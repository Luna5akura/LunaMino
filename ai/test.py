# benchmark.py
import time
from .agent import DQNAgent
from .config import *
from .model import QNet
from .utils import TetrisGame
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def profile_rollout(seed, agent_params, steps=100):
    """测试 CPU 玩游戏并做前向推理的速度"""
    game = TetrisGame(seed=seed)
    local_q_net = QNet()
    local_q_net.load_state_dict(agent_params['q_net'])
    local_q_net.eval()
    
    start_time = time.perf_counter()
    step_count = 0
    
    for _ in range(steps):
        prev_board, prev_ctx = game.get_state()
        moves, previews, ids = game.get_legal_moves_with_previews()
        if len(moves) == 0: break
        
        # 模拟前向推理
        with torch.no_grad():
            post_ctxs = [DQNAgent.approximate_post_ctx(prev_ctx, moves[i], previews[i], prev_board) for i in range(len(previews))]
            post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32)
            boards_t = torch.tensor(previews, dtype=torch.float32)
            q_values = local_q_net(boards_t, post_ctx_t).squeeze()
        
        # 随机选个动作推演
        idx = np.random.randint(len(moves))
        chosen_move = moves[idx]
        game.step(chosen_move[0], chosen_move[1], chosen_move[2], chosen_move[4])
        step_count += 1

    game.close()
    return step_count, time.perf_counter() - start_time

def run_benchmark():
    agent = DQNAgent()
    
    print("="*40)
    print("🚀 开始性能压测 (Benchmark) 🚀")
    print(f"🖥️ 设备: {agent.device}")
    print("="*40)

    # 1. 测试 Rollout (环境采样 + CPU 推理) 性能
    print("\n[1/3] 测试并行环境 Rollout 性能...")
    cpu_state_dict = {k: v.cpu() for k, v in agent.q_net.state_dict().items()}
    agent_params = {'epsilon': 1.0, 'q_net': cpu_state_dict}
    
    total_steps = 0
    start_time = time.perf_counter()
    
    # 模拟多线程收集数据
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures =[executor.submit(profile_rollout, i, agent_params, 200) for i in range(NUM_THREADS)]
        for future in futures:
            steps, _ = future.result()
            total_steps += steps
            
    rollout_time = time.perf_counter() - start_time
    print(f"✅ Rollout 完成: 耗时 {rollout_time:.2f} 秒")
    print(f"⚡ CPU 采样速度: {total_steps / rollout_time:.2f} Steps/s (目标建议: > 2000 Steps/s)")

    # 2. 伪造数据填入 Replay Buffer
    print("\n[2/3] 伪造训练数据填入 Buffer...")
    game = TetrisGame(seed=0)
    _, ctx = game.get_state()
    for _ in range(BATCH_SIZE * 10):
        # 伪造一个 fake transition 存进去
        dummy_moves = np.zeros((10, 5), dtype=np.int8)
        dummy_previews = np.zeros((10, 20, 10), dtype=np.uint8)
        agent.replay_buffer.push(
            dummy_previews[0], ctx, 1.0, dummy_moves, dummy_previews, ctx, False
        )
    game.close()

    # 3. 测试 GPU 训练性能
    print("\n[3/3] 测试 GPU 模型优化性能 (Optimize)...")
    opt_steps = 100
    start_time = time.perf_counter()
    for _ in range(opt_steps):
        agent.optimize()
    train_time = time.perf_counter() - start_time
    
    print(f"✅ Optimize 完成: 耗时 {train_time:.2f} 秒")
    print(f"⚡ GPU 训练速度: {opt_steps / train_time:.2f} Batches/s (目标建议: > 50 Batches/s)")
    print("="*40)
    print("💡 结论分析：如果 CPU 采样速度远低于期望值，说明被 Python GIL 锁死或者 CPU CNN 推理太慢。")
    print("如果 GPU 训练速度慢，说明数据在 CPU->GPU 的搬运成为了瓶颈。")

if __name__ == "__main__":
    run_benchmark()