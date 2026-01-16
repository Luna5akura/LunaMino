# ai/runner.py

from torch import set_num_threads
from torch import manual_seed
import torch.multiprocessing as mp
import numpy as np
import time
import gc
import queue
import random
import os
import threading
from .utils import TetrisGame
from .mcts import MCTS
from .model import TetrisPolicyValue
from .trainer import TetrisTrainer
from .ui import TrainingDashboard
from .reward import calculate_heuristics, get_reward
from . import config
from .inference import InferenceBatcher, RemoteModel, run_inference_loop  # 新增

def select_greedy_move(game):
    # 修改 1: 同时接收 moves 和 ids
    legal_moves, legal_ids = game.get_legal_moves()
    if len(legal_moves) == 0: return None, None
    
    best_score = -float('inf')
    best_move = None
    best_id = -1
    
    # 修改 2: 使用索引遍历，以便同时访问 move 和 id
    for i in range(len(legal_moves)):
        move = legal_moves[i]
        move_id = legal_ids[i]
        
        # Clone 游戏环境进行模拟
        sim_game = game.clone()
        res = sim_game.step(move[0], move[1], move[2], move[4])
        
        board, _ = sim_game.get_state()
        stats = calculate_heuristics(board) 
        
        # 贪婪评分公式
        score = (res[0] * 1000.0) - (stats[1] * 50.0) - (stats[0] * 5.0) - (stats[2] * 2.0)
        
        if score > best_score:
            best_score = score
            best_move = move
            best_id = move_id # 记录对应的 ID
            
        sim_game.close()
        
    return best_move, best_id

def collect_selfplay_data(game, mcts, num_simulations, render=False):
    game.reset(random.randint(0, 1000000))
    mcts.num_simulations = num_simulations
    if render:
        game.enable_render()
   
    episode_data = {'boards': [], 'ctxs': [], 'probs': []}
    score = 0.0
    lines = 0
    steps = 0
    board, prev_ctx = game.get_state()
    h_res = calculate_heuristics(board)
    prev_metrics = h_res[:4] # 前4个是指标
    
    # 探索率：随着步数增加，或者在训练后期，你需要降低这个值
    epsilon = config.GREEDY_EPSILON 
   
    while True:
        if render:
            game.render()

        # ====================================================================
        # [修复重点] 步骤 1: 始终运行 MCTS 获取高质量的训练目标 (Soft Labels)
        # ====================================================================
        # 无论我们稍后决定是否使用贪婪策略，训练数据必须来自 MCTS 的搜索结果。
        # 这样网络学到的是“思考后的概率分布”，而不是极端的“非黑即白”。
        root = mcts.run(game)
        

        max_h = prev_metrics[0]
        
        # [修改] 使用 config 中的参数替换硬编码数字
        if max_h > config.TEMP_EMERGENCY_HEIGHT:
            temp = config.TEMP_EMERGENCY 
        elif steps < config.TEMP_START_STEPS: 
            temp = config.TEMP_HIGH
        else:
            temp = config.TEMP_LOW
            
        mcts_probs = mcts.get_action_probs(root, temp=temp)

        # 记录数据 (始终记录 MCTS 的结果)
        episode_data['boards'].append(board)
        episode_data['ctxs'].append(prev_ctx)
        episode_data['probs'].append(mcts_probs)

        # ====================================================================
        # [修复重点] 步骤 2: 决定执行哪个动作 (Action Selection)
        # ====================================================================
        # Epsilon-Greedy 逻辑只在这里生效，它决定了游戏怎么玩，但不污染 Buffer 数据。
        
        legal_moves, legal_ids = game.get_legal_moves()
        if len(legal_moves) == 0:
            break

        final_move = None
        used_greedy = False

        # --- 分支 A: 尝试贪婪引导 (探索) ---
        if random.random() < epsilon:
            greedy_move, greedy_id = select_greedy_move(game)
            if greedy_move is not None:
                final_move = greedy_move
                used_greedy = True
                # 注意：这里我们只覆盖了动作，没有覆盖上面的 mcts_probs
        
        # --- 分支 B: 如果没有贪婪 (或贪婪失败)，则根据 MCTS 概率采样 ---
        if final_move is None:
            # 过滤无效动作的概率 (保持原逻辑)
            valid_probs = mcts_probs[legal_ids]
            s_valid = valid_probs.sum()
            if s_valid < 1e-9:
                valid_probs = np.ones(len(valid_probs), dtype=np.float32) / len(valid_probs)
            else:
                valid_probs /= s_valid
            
            idx = np.random.choice(len(legal_moves), p=valid_probs)
            final_move = legal_moves[idx]

        #====================================================================
        # 步骤 3: 执行动作并计算奖励
        # ====================================================================
        
        # 调试日志
        if config.DEBUG_MODE and steps % config.DEBUG_FREQ == 0:
             # 验证落点一致性
             valid, _, _, _ = game.validate_step(final_move[0], final_move[1], final_move[2], final_move[4])
             if not valid:
                 print(f"[DEBUG] Step {steps}: Inconsistent! Move: {final_move}")

        # 执行
        res = game.step(final_move[0], final_move[1], final_move[2], final_move[4])
        
        # 状态更新
        next_board, next_ctx = game.get_state()
        h_res_next = calculate_heuristics(next_board)
        cur_metrics = h_res_next[:4]
       
        reward, force_over = get_reward(res, cur_metrics, prev_metrics, steps)
        score += reward
        lines += res[0]
        steps += 1
        
        # 日志区分是谁在发力
        if res[0] > 0:
            if used_greedy:
                print(f"[Greedy] Line Cleared! (+{res[0]}) Total: {lines}")
            else:
                print(f"[Model!] Line Cleared! (+{res[0]}) Total: {lines}") 

        board = next_board
        prev_ctx = next_ctx
        prev_metrics = cur_metrics
       
        if res[3] or force_over or steps > config.MAX_STEPS_TRAIN:
            break
            
    # 结尾处理保持不变
    score = score + config.TANH_OFFSET
    norm_score = np.tanh(score)
    episode_data['values'] = [norm_score] * len(episode_data['boards'])
    
    return episode_data, {'score': score, 'lines': lines, 'steps': steps}


# --- Worker Function (Modified for Remote Inference) ---
def worker_func(rank, shm_name, num_workers, data_queue):
    """
    Worker 现在是一个纯 CPU 数据生成器。
    它不加载模型权重，而是通过 Shared Memory 请求主进程推理。
    """
    # 限制单线程，避免 CPU 争抢
    set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
   
    seed = rank + int(time.time())
    np.random.seed(seed)
    manual_seed(seed)
   
    # 1. 连接共享内存
    batcher = InferenceBatcher(num_workers, local_batch_size=config.MCTS_BATCH_SIZE)
    batcher.connect(shm_name)
   
    # 2. 创建代理模型 (Proxy Model)
    # 这个对象没有权重，只是负责往 shared memory 写数据
    remote_model = RemoteModel(batcher, rank)
   
    # 3. 初始化 MCTS
    # 注意：device='cpu' 即可，因为 tensor 只在 cpu 上流转
    mcts = MCTS(remote_model, device='cpu',
                num_simulations=config.MCTS_SIMS_TRAIN,
                batch_size=config.MCTS_BATCH_SIZE)  # <--- 这里
    game = TetrisGame()

   # === 新增：动态门槛 ===
    # 初期稍微严格一点，强迫它必须比随机乱扔好
    # 随着训练进行，可以动态调整，或者固定一个较低的值
    min_steps_threshold = 25 
    min_lines_threshold = 1

    games = 0
    while True:
        # 收集数据
        data, stats = collect_selfplay_data(game, mcts, config.MCTS_SIMS_TRAIN, render=False)
        
        # === 修改核心：数据门控 (Data Gating) ===
        # 这样 Buffer 里全是"至少做对了一些事"的样本
        is_good_game = (stats['lines'] >= min_lines_threshold) or (stats['steps'] >= min_steps_threshold)
        
        if is_good_game:
            pkg = (
                np.array(data['boards'], dtype=np.int8),
                np.array(data['ctxs'], dtype=np.float32),
                np.array(data['probs'], dtype=np.float16),
                np.array(data['values'], dtype=np.float32),
                stats
            )
           
            try:
                data_queue.put(pkg, timeout=10)
            except queue.Full:
                pass
        else:
            # 可选：如果被丢弃，打印一下（但在多线程下可能会刷屏，建议仅调试用）
            # if rank == 0: print(f"[Worker {rank}] Discarded bad game: Steps={stats['steps']}")
            pass
            
        games += 1

def run_train(reset=False, use_rich=True, workers=16):  # 建议增加 workers 数量
    ui = TrainingDashboard(use_rich=use_rich)
    mode_str = f"Multi(SharedMem, {workers} Workers)"
    ui.log(f"[bold]{mode_str}[/bold]")
    ui.update_stats(mode=mode_str)
   
    mp.set_start_method('spawn', force=True)
   
    # 1. 初始化 Trainer (持有真实模型)
    trainer = TetrisTrainer()
    trainer.game_idx = trainer.load_checkpoint(ui, force_reset=reset)
    # 确保模型在 GPU 且是 eval 模式 (Inference Server 用)
    trainer.model.eval()
   
    # 2. 初始化推理批处理器 (Shared Memory)
    batcher = InferenceBatcher(workers, local_batch_size=config.MCTS_BATCH_SIZE)
    shm_name = batcher.create()
   
    # 3. 启动 Inference Server 线程
    # 使用线程而不是进程，因为它可以直接访问 trainer.model (在 GPU 上)
    stop_event = threading.Event()
    inference_thread = threading.Thread(
        target=run_inference_loop,
        args=(batcher, trainer.model, trainer.device, stop_event),
        daemon=True
    )
    inference_thread.start()
   
    # 4. 启动 Workers
    data_queue = mp.Queue(maxsize=config.QUEUE_MAX_SIZE)  # 队列可以大一点
    procs = []
   
    # 建议将 workers 设置为 CPU 核心数的 1.5倍 到 2倍
    # 因为 Worker 大部分时间在自旋等待 GPU，并不完全占满 CPU 流水线
    print(f"Starting {workers} workers connecting to {shm_name}...")
   
    for i in range(workers):
        p = mp.Process(target=worker_func, args=(i, shm_name, workers, data_queue))
        p.start()
        procs.append(p)
       
    try:
        with ui.get_context() as live:
            while True:
                # -------------------------------------------------
                # 数据收集循环
                # -------------------------------------------------
                new_data_count = 0
                start_time = time.time()
               
                # 快速消费队列
                while new_data_count < 100 and (time.time() - start_time) < 0.5:
                    try:
                        b, c, p, v, stats = data_queue.get_nowait()
                        trainer.buffer.add_batch(b, c, p, v)
                        trainer.game_idx += 1
                        new_data_count += 1
                       
                        ui.update_stats(
                            game_idx=trainer.game_idx,
                            score=stats['score'],
                            lines=stats['lines'],
                            buffer_size=trainer.buffer.size
                        )
                    except queue.Empty:
                        break
               
                # -------------------------------------------------
                # 训练循环 (优化逻辑)
                # -------------------------------------------------
                if trainer.buffer.size >= config.BATCH_SIZE:
                    train_steps = max(1, int(new_data_count * 2))  # 根据你的比例调整
                    train_steps = min(train_steps, 50)
                   
                    # 用于计算平均值
                    avg_metrics = {'loss': 0, 'loss_p': 0, 'loss_v': 0}
                    valid_steps = 0
                   
                    for _ in range(train_steps):
                        metrics = trainer.update_weights()
                        if metrics:
                            for k in avg_metrics:
                                avg_metrics[k] += metrics[k]
                            valid_steps += 1
                   
                    if valid_steps > 0:
                        # 更新 UI
                        ui.update_stats(
                            loss=avg_metrics['loss'] / valid_steps,
                            loss_p=avg_metrics['loss_p'] / valid_steps,
                            loss_v=avg_metrics['loss_v'] / valid_steps
                        )
                        trainer.model.eval()
                # UI Update
                if ui.use_rich:
                    live.update(ui.get_renderable())
                elif new_data_count > 0:
                    # 这里传入 stats 里的 steps (也就是本局存活步数)
                    ui.update_stats(steps=stats['steps'])
                    ui.print_plain()
                           
                # Auto Save
                if trainer.game_idx > 0 and trainer.game_idx % 500 == 0 and new_data_count > 0:
                    trainer.save_checkpoint(ui, backup=True)
                    gc.collect()
               
                if new_data_count == 0:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        ui.log("Stopping...")
        stop_event.set()  # 停止推理线程
        inference_thread.join()
       
        for p in procs:
            p.terminate()
            p.join()
           
        trainer.save_checkpoint(ui)
        batcher.close()
        batcher.unlink()  # 清理共享内存
        print("Cleaned up shared memory.")

def run_gui(reset=False):
    """
    单线程模式，启用游戏渲染界面（通过 utils.py 中的渲染函数）。
    此模式下，直接使用本地模型进行 MCTS 推理，并在 collect_selfplay_data 中启用 render=True 以显示游戏界面。
    """
    ui = TrainingDashboard(use_rich=False)  # 使用 rich UI，如果不需要可设为 False
    mode_str = "Single(GUI)"
    ui.log(f"[bold]{mode_str}[/bold]")
    ui.update_stats(mode=mode_str)

    # 初始化 Trainer 和模型
    trainer = TetrisTrainer()
    trainer.game_idx = trainer.load_checkpoint(ui, force_reset=reset)
    model = trainer.model
    model.eval()
    device = trainer.device

    # 初始化 MCTS，使用本地模型（单线程下可直接用 GPU，如果可用）
    mcts = MCTS(model, device=device, num_simulations=config.MCTS_SIMS_TRAIN, batch_size=1)  # batch_size=1 适合单线程
    game = TetrisGame()

    try:
        with ui.get_context() as live:
            while True:
                # 收集数据，启用渲染（游戏界面）
                episode_data, stats = collect_selfplay_data(game, mcts, config.MCTS_SIMS_TRAIN, render=True)

                # 转换为 numpy 数组并添加到 buffer
                boards = np.array(episode_data['boards'], dtype=np.int8)
                ctxs = np.array(episode_data['ctxs'], dtype=np.float32)
                probs = np.array(episode_data['probs'], dtype=np.float16)
                values = np.array(episode_data['values'], dtype=np.float32)
                trainer.buffer.add_batch(boards, ctxs, probs, values)

                trainer.game_idx += 1

                # 更新 UI 统计
                ui.update_stats(
                    game_idx=trainer.game_idx,
                    score=stats['score'],
                    lines=stats['lines'],
                    steps=stats['steps'],
                    buffer_size=trainer.buffer.size
                )

                # 如果 buffer 足够，进行训练
                if trainer.buffer.size >= config.BATCH_SIZE:
                    metrics = trainer.update_weights()
                    if metrics:
                        ui.update_stats(
                            loss=metrics['loss'],
                            loss_p=metrics['loss_p'],
                            loss_v=metrics['loss_v']
                        )
                    trainer.model.eval()  # 切换回 eval 模式

                # 更新 dashboard
                if ui.use_rich:
                    live.update(ui.get_renderable())
                else:
                    ui.print_plain()

                # 每 100 局保存一次
                if trainer.game_idx % 100 == 0:
                    trainer.save_checkpoint(ui)

                # 短暂休眠，避免 CPU 过载
                time.sleep(0.1)
    except KeyboardInterrupt:
        ui.log("Stopping GUI mode...")
        trainer.save_checkpoint(ui)
        game.close()