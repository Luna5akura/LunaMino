import sys
import os
import numpy as np

# 假设你的目录结构是 Tetris/ai/utils.py，而这个测试脚本在 Tetris/test.py 或 Tetris/ai/test.py
# 确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 如果脚本在 ai/ 目录下，使用: from utils import TetrisGame
# 如果脚本在根目录下，使用: from ai.utils import TetrisGame
try:
    from ai.utils import TetrisGame
except ImportError:
    from utils import TetrisGame

# 测试脚本
def test_engine():
    print("Initializing TetrisGame...")
    game = TetrisGame(seed=42)

    # 1. 测试接收垃圾行
    # 注意：现在的 receive_garbage 逻辑中包含了随机孔位生成
    for i in range(3):
        lines = (i % 3) + 1  # 1-3 lines
        print(f"Receiving {lines} garbage lines...")
        game.receive_garbage(lines)

    # 2. 获取状态并打印简单的 Board 预览
    board, ctx = game.get_state()
    # board是 (20, 10)，索引 0 是底部，19 是顶部 (根据之前的 reshape[::-1] 逻辑，0可能是顶也可能是底，取决于utils的具体实现)
    # 根据之前的 utils 代码：self.np_board = ...reshape(20, 10)[::-1]
    # 这意味着 index 0 是视觉上的顶部 (Y=19)，index 19 是底部 (Y=0)
    print(board) # 打印顶部几行看看是否有垃圾顶上来

    # 3. 获取合法移动
    # 现在的 get_legal_moves 返回的是 numpy int8 数组，形状为 (N, 5)
    # 每一行的格式: [x, y, rotation, landing_height, use_hold]
    moves = game.get_legal_moves()
    
    print(f"\nLegal moves count: {len(moves)}")
    
    if len(moves) > 0:
        print("Sample moves (first 5):")
        # 遍历 numpy 数组
        for i, move in enumerate(moves):
            # 数组解包：基于 utils.py 中的定义顺序
            x, y, rotation, landing_height, use_hold = move
            
            print(f"Move {i}: x={x}, y={y}, rotation={rotation}, "
                  f"landing_height={landing_height}, use_hold={bool(use_hold)}")
            
        # 4. 简单的 Step 测试 (确保返回字典格式正确)
        print("\nExecuting the first legal move...")
        first_move = moves[0]
        # 注意：step 接受的参数依然是分开的整数
        res = game.step(first_move[0], first_move[1], first_move[2], first_move[4])
        
        print(f"Step Result: {res}")
        # 验证返回的是字典 (之前的优化中改成了字典)
        if isinstance(res, dict):
            print("Step verification: Success (Returns Dict)")
            print(f"Lines Cleared: {res['lines_cleared']}")
            print(f"Game Over: {res['game_over']}")
        else:
            print(f"Step verification: Warning (Returned {type(res)}, expected Dict)")

    else:
        print("No legal moves available (Game Over or Bug).")

if __name__ == "__main__":
    test_engine()