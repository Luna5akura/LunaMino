# ai/runner.py

import torch
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
from .inference import InferenceBatcher, RemoteModel, run_inference_loop # 新增

# ... (collect_selfplay_data 函数保持不变，不需要改动) ...
def collect_selfplay_data(game, mcts, num_simulations, render=False):
    # 保持原样...
    # (为了节省篇幅，这里省略 collect_selfplay_data 的代码，请保留你现有的版本)
    # ...
    game.reset(random.randint(0, 1000000))
    mcts.num_simulations = num_simulations
    if render: game.enable_render()
    
    episode_data = {'boards': [], 'ctxs': [], 'probs': []}
    score = 0.0; lines = 0; steps = 0
    board, prev_ctx = game.get_state()
    prev_metrics = calculate_heuristics(board)
    
    while True:
        # 减少打印频率
        if render or steps % 50 == 0: 
            pass # print debug if needed

        if render: game.render()
        root = mcts.run(game) # 这里 mcts 内部会调用 RemoteModel
        
        # ... (后续逻辑不变) ...
        max_h = prev_metrics[0]
        if max_h > 14: temp = 0.1
        elif steps < 30: temp = 1.0
        else: temp = 0.5
        
        action_probs = mcts.get_action_probs(root, temp=temp)
        episode_data['boards'].append(board)
        episode_data['ctxs'].append(prev_ctx)
        episode_data['probs'].append(action_probs)
        
        legal, ids = game.get_legal_moves()
        if len(legal) == 0: break
        
        valid_probs = action_probs[ids]
        s_valid = valid_probs.sum()
        if s_valid < 1e-9: valid_probs = np.ones(len(valid_probs), dtype=np.float32)/len(valid_probs)
        else: valid_probs /= s_valid
        
        idx = np.random.choice(len(legal), p=valid_probs)
        move = legal[idx]
        res = game.step(move[0], move[1], move[2], move[4])
        
        next_board, next_ctx = game.get_state()
        cur_metrics = calculate_heuristics(next_board)
        
        # Tuple 解包
        reward, force_over = get_reward(res, cur_metrics, prev_metrics, steps)
        score += reward; lines += res[0]; steps += 1
        board = next_board; prev_ctx = next_ctx; prev_metrics = cur_metrics
        
        if res[3] or force_over or steps > config.MAX_STEPS_TRAIN: break
            
    norm_score = np.tanh(score / 20.0)
    episode_data['values'] = [norm_score] * len(episode_data['boards'])
    return episode_data, {'score': score, 'lines': lines, 'steps': steps}

# --- Worker Function (Modified for Remote Inference) ---
def worker_func(rank, shm_name, num_workers, data_queue):
    """
    Worker 现在是一个纯 CPU 数据生成器。
    它不加载模型权重，而是通过 Shared Memory 请求主进程推理。
    """
    # 限制单线程，避免 CPU 争抢
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    
    seed = rank + int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. 连接共享内存
    batcher = InferenceBatcher(num_workers)
    batcher.connect(shm_name)
    
    # 2. 创建代理模型 (Proxy Model)
    # 这个对象没有权重，只是负责往 shared memory 写数据
    remote_model = RemoteModel(batcher, rank)
    
    # 3. 初始化 MCTS
    # 注意：device='cpu' 即可，因为 tensor 只在 cpu 上流转
    mcts = MCTS(remote_model, device='cpu', 
                num_simulations=config.MCTS_SIMS_TRAIN,
                batch_size=config.BATCH_SIZE)
    game = TetrisGame()
    
    games = 0
    while True:
        # 收集数据
        # print(f"Worker {rank} starting game {games}")
        data, stats = collect_selfplay_data(game, mcts, config.MCTS_SIMS_TRAIN, render=False)
        
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
            pass # 队列满就丢弃，或者等一会，防止 Worker 挂起太久
            
        games += 1

def run_train(reset=False, use_rich=True, workers=16): # 建议增加 workers 数量
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
    batcher = InferenceBatcher(workers)
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
    data_queue = mp.Queue(maxsize=256) # 队列可以大一点
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
                # 训练循环 (Training Step)
                # -------------------------------------------------
                if trainer.buffer.size >= config.BATCH_SIZE:
                    # 暂停 Inference Server 吗？
                    # 不，我们希望边训练边推演。PyTorch 可以在同一 GPU 上并发执行 Kernel。
                    # 但为了防止显存冲突，也可以加锁，不过通常不需要。
                    
                    # 动态调整训练步数
                    train_steps = max(1, int(new_data_count * 1.0))
                    train_steps = min(train_steps, 20)
                    
                    total_loss = 0
                    for _ in range(train_steps):
                        total_loss += trainer.update_weights()
                    
                    if train_steps > 0:
                        ui.update_stats(loss=total_loss/train_steps)
                        
                        # 重要：Inference Server 使用的是 trainer.model 的引用
                        # 训练更新了权重，Inference Server 自动就会用到新权重
                        # 不需要 load_state_dict 同步！这是线程架构的巨大优势。
                        trainer.model.eval() # 训练完切回 eval 模式供推演使用

                # UI Update
                if ui.use_rich: live.update(ui.get_renderable())
                elif new_data_count > 0: ui.print_plain()
                
                # Auto Save
                if trainer.game_idx > 0 and trainer.game_idx % 500 == 0 and new_data_count > 0:
                    trainer.save_checkpoint(ui, backup=True)
                    gc.collect()
                
                if new_data_count == 0:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        ui.log("Stopping...")
        stop_event.set() # 停止推理线程
        inference_thread.join()
        
        for p in procs: 
            p.terminate()
            p.join()
            
        trainer.save_checkpoint(ui)
        batcher.close()
        batcher.unlink() # 清理共享内存
        print("Cleaned up shared memory.")