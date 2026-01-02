# ai/test/test.py

import sys
import os
import time
import numpy as np
import torch

# 确保能导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from ai.utils import TetrisGame
from ai.model import TetrisPolicyValue
from ai.mcts import MCTS
from ai.buffer import NumpyReplayBuffer
from ai.trainer import TetrisTrainer
from ai.reward import calculate_heuristics, get_reward
from ai import config
from ai.config import ACTION_DIM

def print_header(title):
    print(f"\n{'='*15} {title} {'='*15}")

# ==========================================
# 1. 基础 C++ 接口与内存测试
# ==========================================
def test_memory_layout(game):
    print_header("1. 内存布局与连续性检查")
    board, ctx = game.get_state()
    
    print(f"Board Shape: {board.shape} (Expect: (20, 10))")
    print(f"Context Shape: {ctx.shape} (Expect: (11,))")
    
    is_board_contiguous = board.flags['C_CONTIGUOUS']
    print(f"Board Contiguous? {is_board_contiguous}")
    
    if not is_board_contiguous:
        print("❌ 失败: Board 内存不连续，会导致 C++ 读取错误。")
        return False
    print("✅ 内存检查通过")
    return True

def test_action_alignment(game):
    print_header("2. 动作数据结构对齐检查")
    moves, ids = game.get_legal_moves()
    count = len(ids)
    print(f"生成动作数量: {count}")
    
    if count == 0: return True

    # 检查 Struct 字段
    xs = moves[:, 0]
    ys = moves[:, 1] # landing height
    
    if np.any(xs > 9) or np.any(xs < -2):
        print(f"❌ 失败: X 坐标异常 {xs[np.where((xs>9)|(xs<-2))]}，Struct 对齐错误！")
        return False
        
    if np.any(ids < 0) or np.any(ids >= ACTION_DIM):
        print(f"❌ 失败: ID 异常范围 [{ids.min()}, {ids.max()}]")
        return False

    print("✅ 对齐检查通过")
    return True

def test_game_logic(game):
    print_header("3. 游戏 Step 接口测试")
    moves, _ = game.get_legal_moves()
    if len(moves) == 0: return False
    
    action = moves[0]
    # step 返回的是 tuple: (lines, damage, type, game_over, b2b, combo)
    res = game.step(action[0], action[1], action[2], action[4])
    
    print(f"Step Result (Tuple): {res}")
    
    if not isinstance(res, tuple):
        print(f"❌ 失败: Step 应返回 tuple，实际返回 {type(res)}")
        return False

    # 检查元组长度 (C++ StepResultStruct 有 6 个字段)
    if len(res) != 6:
        print(f"❌ 失败: Step 返回元组长度不对，期望 6，实际 {len(res)}")
        return False
        
    print("✅ 逻辑检查通过")
    return True

# ==========================================
# 2. 奖励与启发式测试 (新增)
# ==========================================
def test_reward_system(game):
    print_header("4. 奖励与启发式计算测试")
    
    board, _ = game.get_state()
    
    # 1. 测试 calculate_heuristics 返回的是否是 tuple
    start_t = time.time()
    heuristics = calculate_heuristics(board)
    duration = time.time() - start_t
    
    print(f"Heuristics: {heuristics}")
    print(f"Time: {duration*1000:.4f} ms")
    
    if not isinstance(heuristics, tuple):
        print(f"❌ 失败: calculate_heuristics 应返回 tuple，实际返回 {type(heuristics)}")
        return False
        
    if len(heuristics) != 4:
        print(f"❌ 失败: Heuristics 元组长度应为 4 (max_h, holes, bump, agg)，实际 {len(heuristics)}")
        return False

    # 2. 测试 get_reward 接口
    # 构造假数据
    dummy_step_res = (0, 0, 0, False, 0, 0) # lines, damage, type, over, b2b, combo
    prev_heuristics = (0, 0, 0, 0)
    
    try:
        reward, force_over = get_reward(dummy_step_res, heuristics, prev_heuristics, steps_survived=10)
        print(f"Reward: {reward}, ForceOver: {force_over}")
    except Exception as e:
        print(f"❌ 失败: get_reward 调用崩溃: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✅ 奖励系统测试通过")
    return True

# ==========================================
# 3. 模型与 MCTS 测试
# ==========================================
def test_model_and_mcts():
    print_header("5. Model & MCTS 集成测试")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = TetrisPolicyValue().to(device)
        model.eval()
        
        # 测试 Forward
        dummy_board = torch.zeros(2, 1, 20, 10).to(device)
        dummy_ctx = torch.zeros(2, 11).to(device)
        with torch.inference_mode():
            logits, vals = model(dummy_board, dummy_ctx)
            
        print(f"Model Out: Logits {logits.shape}, Values {vals.shape}")
        
        # 测试 MCTS
        game = TetrisGame(seed=999)
        mcts = MCTS(model, device=device, num_simulations=50, batch_size=8)
        
        start_t = time.time()
        root = mcts.run(game)
        duration = time.time() - start_t
        
        print(f"MCTS Run Time: {duration:.4f}s")
        probs = mcts.get_action_probs(root)
        
        if np.sum(probs) < 0.99:
            print(f"❌ 失败: MCTS 概率和不为 1 ({np.sum(probs)})")
            return
            
        print("✅ MCTS 测试通过")
        
    except Exception as e:
        print(f"❌ MCTS/Model 失败: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 4. Buffer & Trainer 测试 (类型修正)
# ==========================================
def test_training_pipeline():
    print_header("6. Buffer & Trainer (Int8/FP16) 测试")
    
    try:
        buffer = NumpyReplayBuffer(capacity=100)
        
        # 1. 构造符合优化后 Buffer 要求的数据 (Int8, Float16)
        # 注意：runner.py 中是在 collect 之后转成 numpy 数组放入 queue 的
        # 这里模拟 queue 取出来的数据
        n = 10
        b = np.random.randint(0, 2, size=(n, 20, 10)).astype(np.int8) # Int8!
        c = np.random.randn(n, 11).astype(np.float32)
        p = np.random.rand(n, config.ACTION_DIM).astype(np.float16)   # Float16!
        v = np.random.rand(n).astype(np.float32)
        
        buffer.add_batch(b, c, p, v)
        print(f"Buffer stored {buffer.size} items.")
        
        # 2. 验证 sample 返回类型
        s_b, s_c, s_p, s_v = buffer.sample(4)
        print(f"Sampled Board Dtype: {s_b.dtype} (Expect int8)")
        print(f"Sampled Probs Dtype: {s_p.dtype} (Expect float16)")
        
        if s_b.dtype != np.int8:
            print("❌ 失败: Buffer 采样 Board 应保持 int8 以节省带宽")
            return
        if s_p.dtype != np.float16:
            print("❌ 失败: Buffer 采样 Probs 应保持 float16")
            return

        # 3. 测试 Trainer (GPU 类型转换)
        trainer = TetrisTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 填满 batch
        while trainer.buffer.size < config.BATCH_SIZE:
            trainer.buffer.add_batch(b, c, p, v)
            
        loss = trainer.update_weights()
        print(f"Trainer Update Loss: {loss:.4f}")
        
        print("✅ Trainer 流程测试通过")

    except Exception as e:
        print(f"❌ Trainer 测试失败: {e}")
        import traceback
        traceback.print_exc()

# ai/test/test.py 中的新增部分

def test_mcts_speed_benchmark():
    print_header("7. MCTS 速度基准测试 (Speed Benchmark)")
    
    # 1. 准备环境
    # 注意：模拟真实 Worker 环境。如果 Worker 用 CPU，这里也该测 CPU。
    # 通常 config.DEVICE 是 cuda，但在多进程 spawn 中，Worker 里的模型往往跑在 CPU 上（除非你做了特殊处理）。
    # 这里我们测试 config.DEVICE 定义的设备。
    device = config.DEVICE 
    print(f"Testing on Device: {device}")
    
    model = TetrisPolicyValue().to(device)
    model.eval()
    
    # 2. 获取当前配置的模拟次数
    sim_count = config.MCTS_SIMS_TRAIN
    print(f"Current Config MCTS_SIMS_TRAIN: {sim_count}")
    
    game = TetrisGame(seed=999)
    # 初始化 MCTS
    mcts = MCTS(model, device=device, num_simulations=sim_count)
    
    # 3. 预热 (Warmup)
    # 这一步非常重要！第一次运行包含了 Numba JIT 编译、Cuda 初始化、内存分配等开销。
    print("正在预热 (Warmup)... (耗时较长是正常的)")
    t0 = time.time()
    mcts.run(game) 
    print(f"预热耗时: {time.time() - t0:.4f}s")
    
    # 4. 正式测试 (运行 5 步取平均值)
    steps_to_test = 5
    total_time = 0.0
    
    print(f"开始测试 (运行 {steps_to_test} 步)...")
    
    for i in range(steps_to_test):
        start = time.time()
        # 运行 MCTS 思考一步
        root = mcts.run(game) 
        dt = time.time() - start
        total_time += dt
        
        # 为了更真实，执行一步
        action_probs = mcts.get_action_probs(root)
        legal, ids = game.get_legal_moves()
        # 简单选概率最高的
        idx = np.argmax(action_probs[ids]) 
        move = legal[idx]
        game.step(move[0], move[1], move[2], move[4])
        
        print(f"  Step {i+1}: {dt:.4f}s")

    avg_time = total_time / steps_to_test
    
    # 5. 结果分析与估算
    print(f"\n📊 统计结果:")
    print(f"  平均每步耗时 (Time per Step): {avg_time:.4f} 秒")
    print(f"  每秒模拟次数 (Simulations/sec): {sim_count / avg_time:.1f}")
    
    # 估算一局游戏时间 (假设一局玩 100 步)
    est_game_steps = 100
    est_total_time = avg_time * est_game_steps
    
    print(f"🔮 估算一局游戏 (100步) 耗时: {est_total_time:.1f} 秒 ({est_total_time/60:.1f} 分钟)")
    
    if avg_time > 0.5:
        print("\n[结论] 🐢 速度较慢: 这解释了为什么看起来像'卡住'了。")
        print("         在没有任何输出的情况下，几十秒没有反应是正常的。")
        print("         建议: 在 config.py 中将 MCTS_SIMS_TRAIN 调低 (例如 50) 用于调试。")
    else:
        print("\n[结论] 🐇 速度很快: 如果程序依然卡住不动，可能是多进程死锁问题。")



def main():
    game = TetrisGame(seed=42)
    
    if not test_memory_layout(game): return
    if not test_action_alignment(game): return
    if not test_game_logic(game): return
    if not test_reward_system(game): return # 新增
    
    game.close()
    
    test_model_and_mcts()
    test_training_pipeline()
    test_mcts_speed_benchmark()
    
    print("\n🎉 所有关键模块测试通过！")

if __name__ == "__main__":
    main()