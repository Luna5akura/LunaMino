# /ai/play.py

import time
import argparse
import torch
import numpy as np
from .agent import DQNAgent
from .utils import TetrisGame

def watch_ai_play(model_path, step_delay):
    print(f"🎮 正在加载模型: {model_path}")
    
    # 1. 初始化 Agent 并加载权重
    agent = DQNAgent()
    try:
        agent.load_checkpoint(model_path)
    except FileNotFoundError:
        print(f"❌ 找不到模型文件 {model_path}，请确认你是否已经开始训练。")
        return

    # 2. 开启纯贪心模式 (无随机探索)
    agent.epsilon = 0.0 
    agent.q_net.eval()

    # 3. 初始化游戏并开启渲染
    game = TetrisGame(seed=int(time.time()))
    game.enable_render()

    total_lines = 0
    is_game_over = False

    print(f"📺 渲染已启动！(动作间隔: {step_delay}秒)")
    print("💡 提示：在终端按下 Ctrl+C 即可退出观战。")

    try:
        while not is_game_over:
            # --- 阶段 A：AI 思考与行动 ---
            prev_board, prev_ctx = game.get_state()
            moves, previews, ids = game.get_legal_moves_with_previews()

            if len(moves) == 0:
                break

            # 为了精确预测消行，直接在这里手写前向推理 (传递 prev_board)
            with torch.no_grad():
                post_ctxs =[DQNAgent.approximate_post_ctx(prev_ctx, moves[i], previews[i], prev_board) for i in range(len(previews))]
                post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32).to(agent.device)
                boards_t = torch.tensor(previews, dtype=torch.float32).to(agent.device)
                q_values = agent.q_net(boards_t, post_ctx_t).squeeze()
                idx = torch.argmax(q_values).item() if q_values.dim() > 0 else 0
            
            # 执行最佳动作
            chosen_move = moves[idx]
            x, y, rot, land, hold = chosen_move
            step_res = game.step(x, y, rot, hold)
            
            total_lines += step_res[0]
            is_game_over = step_res[3]

            print(f"\r🌟 当前消除行数: {total_lines}", end="", flush=True)

            # --- 阶段 B：渲染与画面停留 ---
            # 因为 C 端配置了 60FPS，每次 render 会阻塞约 1/60 秒
            # 我们通过循环调用 render 来充当延迟，同时保证窗口不卡死
            frames_to_wait = int(step_delay * 60)
            if frames_to_wait <= 0:
                frames_to_wait = 1 # 至少渲染1帧
                
            for _ in range(frames_to_wait):
                game.render()

        # 游戏结束后，再停留 3 秒钟让你看清死局的画面
        print("\n💀 游戏结束！展示最终画面 3 秒...")
        for _ in range(180): # 60 * 3 = 180 帧
            game.render()

    except KeyboardInterrupt:
        print("\n🛑 用户强制中止了观战。")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        game.close()
        print(f"🏁 最终成绩 - 消除行数: {total_lines}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch Tetris AI Play!")
    parser.add_argument("--model", type=str, default="checkpoints/best_tetris_ai.pth", 
                        help="你想查看的模型文件路径")
    parser.add_argument("--delay", type=float, default=0.1, 
                        help="AI 每次落子的延迟时间(秒)，默认0.1秒。数值越小下得越快。")
    args = parser.parse_args()
    
    watch_ai_play(args.model, args.delay)