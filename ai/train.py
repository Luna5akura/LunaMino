# ai/train.py

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp

# 限制 PyTorch 的 CPU 线程冲突
torch.set_num_threads(1)
# 开启 RTX 30 系列 Tensor Core 加速
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from .agent import DQNAgent
from .model import QNet
from .config import *
from .utils import TetrisGame

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "tetris_ai.pth")
BEST_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "best_tetris_ai.pth")
if not os.path.exists(CHECKPOINT_DIR): 
    os.makedirs(CHECKPOINT_DIR)

def rollout_worker(worker_id, result_queue, shared_info):
    """
    独立的工作进程（生产者）：负责不断玩游戏、收集经验，然后通过队列发给主进程。
    """
    # 给每个进程设置不同的随机种子，防止所有的 AI 都在下同一盘棋
    np.random.seed(int(time.time()) + worker_id * 100)
    torch.manual_seed(int(time.time()) + worker_id * 100)
    
    game = TetrisGame(seed=int(time.time()) + worker_id)
    local_q_net = QNet()
    
    while True:
        # 接收主进程的停止信号
        if shared_info['stop']:
            break
            
        # 1. 从共享内存中获取最新的模型权重和 Epsilon
        local_q_net.load_state_dict(shared_info['q_net_state'])
        local_q_net.eval()
        local_epsilon = shared_info['epsilon']

        total_reward = 0
        total_lines = 0
        transitions =[]
        
        # 2. 开始玩一把游戏
        for step_num in range(MAX_STEPS_PER_EPISODE):
            prev_board, prev_ctx = game.get_state()
            moves, previews, ids = game.get_legal_moves_with_previews()
            
            if len(moves) == 0:
                break

            # Epsilon-Greedy 策略
            if np.random.rand() < local_epsilon:
                idx = np.random.randint(len(moves))
            else:
                with torch.no_grad():
                    # 在纯 CPU 上进行前向推理
                    post_ctxs = [DQNAgent.approximate_post_ctx(prev_ctx, moves[i], previews[i], prev_board) for i in range(len(previews))]
                    post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32)
                    boards_t = torch.tensor(previews, dtype=torch.float32)
                    q_values = local_q_net(boards_t, post_ctx_t).squeeze()
                    if q_values.dim() == 0:
                        idx = 0
                    else:
                        idx = torch.argmax(q_values).item()
                        
            chosen_move = moves[idx]
            chosen_preview = previews[idx]
            x, y, rot, land, hold = chosen_move
            
            # 执行动作
            step_res = game.step(x, y, rot, hold)
            post_board, post_ctx = game.get_state()
            done = step_res[3]
            total_lines += step_res[0]
            
            reward = compute_reward(step_res, prev_board, post_board, land)
            
            MAX_EVALS = 20
            next_previews = np.zeros((MAX_EVALS, 20, 10), dtype=np.uint8)
            next_ctxs = np.zeros((MAX_EVALS, 11), dtype=np.float32)
            
            if not done:
                moves_next, previews_next, _ = game.get_legal_moves_with_previews()
                count = len(previews_next)
                
                if count > 0:
                    indices = np.random.choice(count, MAX_EVALS, replace=(count < MAX_EVALS))
                    sampled_moves = moves_next[indices]
                    # 覆盖掉前面的全零矩阵
                    next_previews = previews_next[indices]
                    
                    post_ctxs =[DQNAgent.approximate_post_ctx(post_ctx, sampled_moves[j], next_previews[j], post_board) for j in range(MAX_EVALS)]
                    next_ctxs = np.array(post_ctxs, dtype=np.float32)
                else:
                    done = True

            transitions.append((chosen_preview, post_ctx, reward, next_previews, next_ctxs, None, done))
            total_reward += reward
            
            if done:
                break
                
        # 游戏结束，重置环境
        game.reset(seed=int(time.time()) + worker_id + np.random.randint(10000))
        
        # 3. 将这局游戏的成果发送给主进程的队列
        try:
            result_queue.put((worker_id, transitions, total_reward, total_lines))
        except Exception:
            break

    game.close()

def train():
    """
    主进程（消费者）：负责收集数据、放入 Buffer 并驱动 GPU 进行网络优化。
    """
    # PyTorch 推荐在多进程环境使用 spawn 模式，避免 CUDA 初始化冲突
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    
    agent = DQNAgent()
    try:
        agent.load_checkpoint(CHECKPOINT_FILE)
        print(f"Loaded checkpoint. Steps: {agent.steps}, Epsilon: {agent.epsilon:.3f}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Starting fresh.")
        
    # 【新增】：在加载完干净的 Checkpoint 后，再对主进程的模型进行底层编译加速
    if hasattr(torch, 'compile') and agent.device.type == 'cuda':
        print("⚡ 检测到 CUDA，正在启动 Triton 编译器进行底层加速 (首次运行可能需要等待几秒)...")
        agent.q_net = torch.compile(agent.q_net)
        agent.target_net = torch.compile(agent.target_net)

    print("🚀 Starting Asynchronous DQN Training...")

    # 设置一个容量上限防止内存爆炸
    result_queue = mp.Queue(maxsize=NUM_THREADS * 3)
    
    # 共享字典，用于向子进程广播最新模型权重
    shared_info = manager.dict()
    cpu_state_dict = {k.replace('_orig_mod.', ''): v.cpu() for k, v in agent.q_net.state_dict().items()}
    shared_info['q_net_state'] = cpu_state_dict
    shared_info['epsilon'] = agent.epsilon
    shared_info['stop'] = False

    # 启动工作进程
    workers =[]
    for i in range(NUM_THREADS):
        p = mp.Process(target=rollout_worker, args=(i, result_queue, shared_info))
        p.daemon = True
        p.start()
        workers.append(p)

    episode = 0
    last_weight_update_time = time.time()
    
    try:
        # ---- 性能统计变量 ----
        total_wait_time = 0.0
        total_opt_time = 0.0
        total_sync_time = 0.0
        stats_episodes = 0
        
        while episode < MAX_EPISODES:
            # 1. 测算【等待数据耗时】 (Wait Time)
            t0 = time.time()
            worker_id, trans_list, ep_reward, ep_lines = result_queue.get()
            t_wait = time.time() - t0
            
            for trans in trans_list:
                agent.replay_buffer.push(*trans)
                
            opt_steps = max(1, len(trans_list) // 4)
            opt_steps = min(opt_steps, 50)
            total_loss = 0
            actual_opt_steps = 0
            
            # 2. 测算【GPU 训练耗时】 (Optimize Time)
            t1 = time.time()
            for _ in range(opt_steps):
                loss = agent.optimize()
                if loss != 0.0:
                    total_loss += loss
                    actual_opt_steps += 1
            t_opt = time.time() - t1
            
            avg_loss = total_loss / actual_opt_steps if actual_opt_steps > 0 else 0.0
            
            # 3. 测算【权重同步耗时】 (Sync Time)
            t2 = time.time()
            current_time = time.time()
            if current_time - last_weight_update_time > 3.0 or episode % 10 == 0:
                cpu_state_dict = {k.replace('_orig_mod.', ''): v.cpu() for k, v in agent.q_net.state_dict().items()}
                shared_info['q_net_state'] = cpu_state_dict
                shared_info['epsilon'] = agent.epsilon
                last_weight_update_time = current_time
            t_sync = time.time() - t2

            # 累计时间
            total_wait_time += t_wait
            total_opt_time += t_opt
            total_sync_time += t_sync
            stats_episodes += 1
            episode += 1

            # ---- 打印性能诊断面板（每 10 局触发一次） ----
            if stats_episodes >= 10:
                q_size = result_queue.qsize() if hasattr(result_queue, 'qsize') else -1
                total_time = total_wait_time + total_opt_time + total_sync_time
                
                print("\n" + "="*50)
                print(f"📊[性能诊断面板 | 过去10局统计]")
                print(f"📦 队列拥挤度: {q_size} / {result_queue._maxsize}")
                print(f"⏳ 等待Worker产生数据耗时: {total_wait_time:.2f}s (占比 {total_wait_time/total_time:.1%})")
                print(f"🔥 GPU 训练网络耗时:       {total_opt_time:.2f}s (占比 {total_opt_time/total_time:.1%})")
                print(f"🔄 跨进程同步权重耗时:     {total_sync_time:.2f}s (占比 {total_sync_time/total_time:.1%})")
                print("="*50 + "\n")
                
                # 重置统计
                total_wait_time = total_opt_time = total_sync_time = 0.0
                stats_episodes = 0

            # 原本的正常打印
            print(f"Ep {episode:04d} | Wkr {worker_id} | R:{ep_reward:6.1f} | Lns:{ep_lines:3d} | Loss:{avg_loss:6.4f} | Eps:{agent.epsilon:.3f} | Buf:{len(agent.replay_buffer)}")
            
            # ... 下面保留你原来的 agent.save_checkpoint 逻辑 ...
            if ep_reward > agent.best_reward:
                agent.best_reward = ep_reward
                print(f"🌟 --> New Best: {ep_reward:.1f}")
                agent.save_checkpoint(BEST_MODEL_FILE)
            if episode > 0 and episode % 50 == 0:
                agent.save_checkpoint(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\n🛑 训练被用户主动中断。")
    finally:
        print("🧹 正在清理并关闭工作进程...")
        shared_info['stop'] = True
        
        # 必须先清空队列，否则子进程的 put() 会死锁导致无法退出
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break
                
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        print("✅ 关闭完成，训练结束。")

if __name__ == "__main__":
    train()