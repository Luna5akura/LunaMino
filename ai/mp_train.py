import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import os
import pickle
import time
from collections import deque
from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS

# --- 配置 ---
NUM_WORKERS = 10          # 你的 CPU 核心数 (建议设置为 核心数 - 2)
MCTS_SIMS = 30           # 保持较小的搜索次数以保证速度
BATCH_SIZE = 128         # 训练批次
MEMORY_SIZE = 20000      # 总经验池大小
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 文件路径
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"

def worker_process(rank, shared_model, data_queue, device):
    """
    工作进程：负责一直在玩游戏，产生数据并推送到队列
    """
    # 每个进程必须设置随机种子，否则所有进程玩的一样
    torch.manual_seed(rank + int(time.time()))
    np.random.seed(rank + int(time.time()))
    random.seed(rank + int(time.time()))
    
    # 初始化环境 (注意：TetrisGame 必须在进程内初始化，不能跨进程传递)
    try:
        game = TetrisGame()
    except Exception as e:
        print(f"[Worker {rank}] Failed to init game: {e}")
        return

    # 初始化 MCTS
    # 注意：这里我们使用 shared_model，它会自动同步主进程的权重更新
    mcts = MCTS(shared_model, device=device, num_simulations=MCTS_SIMS)
    
    print(f"[Worker {rank}] Started. Device: {device}")
    
    while True:
        # 重置游戏
        lib_ptr = game.ptr # 记录一下，防止重置逻辑出错
        # 这里的 reset 是隐式的，当游戏结束 loop break 后，我们会重新 run loop
        # 但我们需要手动重置 game state 吗？
        # TetrisGame 在 init 时会 reset。
        # 简单起见，我们在 loop 里 check game_over，如果 over 了就自动 restart
        # 但 utils.py 里的实现是单次初始化的。
        # 我们需要在 utils.py 里加一个 reset 接口，或者这里重新 new 一个
        # 为了性能，我们在 C 里加了 ai_reset_game，这里直接调用
        
        # 每次游戏开始前重置
        from .utils import lib
        seed = random.randint(0, 1000000)
        lib.ai_reset_game(game.ptr, seed)
        
        steps = 0
        game_data = []
        total_score = 0
        
        while True:
            # MCTS 搜索
            root = mcts.run(game)
            
            # 温度系数
            temp = 1.0 if steps < 20 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
            
            # 收集状态
            board, ctx, p_type = game.get_state()
            # 保存样本: [board, ctx, p_type, probs, value_placeholder]
            game_data.append([board, ctx, p_type, action_probs, None])
            
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            
            # 解码
            use_hold = 0
            if action_idx >= 40:
                use_hold = 1
                action_idx -= 40 # 变回 0-39
                
            x = action_idx // 4
            rot = action_idx % 4
            
            # 执行
            res = game.step(x, rot, use_hold)
            
            # 奖励计算
            step_reward = 0.01
            if res['lines_cleared'] > 0:
                step_reward += res['lines_cleared'] * 0.2
            if res['damage_sent'] > 0:
                step_reward += res['damage_sent'] * 1.0
            if res['game_over']:
                step_reward -= 1.0
            
            total_score += step_reward
            steps += 1
            
            if res['game_over'] or steps > 400:
                break
        
        # 游戏结束，回填 Value
        final_val = total_score
        if final_val > 10: final_val = 1.0
        elif final_val < -1: final_val = -1.0
        else: final_val = final_val / 10.0
        
        # 将本局数据推送到队列
        for item in game_data:
            item[4] = final_val
            # 必须把 numpy 转为纯数据或 tensor 放入队列，避免共享内存泄漏
            # 这里 item 里的已经是 numpy/list，直接放即可
            pass
            
        # 发送数据给主进程 (阻塞式发送，防止队列溢出)
        # 为了减少通信开销，一次发送整个列表
        data_queue.put(game_data)
        
        # 可以在这里打印一下 worker 的进度
        print(f"[Worker {rank}] Finished game. Score: {total_score:.2f}")

def save_checkpoint(net, optimizer, game_cnt):
    print(f"\n[Main] Saving checkpoint to {CHECKPOINT_FILE}...")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_cnt
    }, CHECKPOINT_FILE)

def train_manager():
    # 1. 必须设置启动方法为 spawn，否则 ctypes 会在 fork 时崩溃
    mp.set_start_method('spawn', force=True)
    
    # 2. 初始化模型并放入共享内存
    shared_model = TetrisNet().to(DEVICE)
    shared_model.share_memory() # 关键：允许跨进程共享权重
    
    optimizer = optim.Adam(shared_model.parameters(), lr=0.001)
    
    # 加载断点
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        print(f"[Main] Loading checkpoint {CHECKPOINT_FILE}...")
        ckpt = torch.load(CHECKPOINT_FILE)
        shared_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_idx = ckpt['game_idx']

    # 3. 创建数据队列和进程
    data_queue = mp.Queue(maxsize=100) # 限制队列大小，防止内存爆
    processes = []
    
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(rank, shared_model, data_queue, DEVICE))
        p.start()
        processes.append(p)
        
    print(f"[Main] Started {NUM_WORKERS} worker processes.")
    
    # 4. 主循环：收集数据 -> 训练
    memory = deque(maxlen=MEMORY_SIZE)
    # 如果有旧的 memory 文件也可以加载... (略)
    
    game_cnt = start_idx
    total_samples = 0
    
    try:
        while True:
            # 从队列获取数据
            # 只要队列不空，就一直取，直到凑够一个 Batch 或者取空
            while not data_queue.empty():
                try:
                    new_games_data = data_queue.get_nowait()
                    memory.extend(new_games_data)
                    game_cnt += 1
                    total_samples += len(new_games_data)
                    
                    if game_cnt % 10 == 0:
                        print(f"[Main] Collected Game {game_cnt}. Memory: {len(memory)}")
                except:
                    break
            
            # 如果经验不够，稍微等一下让 Workers 跑
            if len(memory) < BATCH_SIZE:
                time.sleep(1)
                continue
                
            # 训练步
            batch = random.sample(memory, BATCH_SIZE)
            
            # 转换数据
            b_board = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_ctx = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_ptype = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.long).to(DEVICE)
            b_policy = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_value = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            p_logits, v_pred = shared_model(b_board, b_ctx, b_ptype)
            
            log_probs = F.log_softmax(p_logits, dim=1)
            policy_loss = -torch.sum(b_policy * log_probs, dim=1).mean()
            value_loss = F.mse_loss(v_pred, b_value)
            
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            # 打印 Loss
            if total_samples % 1000 < BATCH_SIZE: # 偶尔打印
                print(f"[Main] Train Step. Loss: {loss.item():.4f}")
                
            # 定期保存
            if game_cnt % 100 == 0 and game_cnt > start_idx:
                save_checkpoint(shared_model, optimizer, game_cnt)
                start_idx = game_cnt # 防止重复保存

    except KeyboardInterrupt:
        print("[Main] Ctrl+C detected. Terminating workers...")
        for p in processes:
            p.terminate()
            p.join()
        save_checkpoint(shared_model, optimizer, game_cnt)
        print("[Main] Done.")

if __name__ == "__main__":
    train_manager()