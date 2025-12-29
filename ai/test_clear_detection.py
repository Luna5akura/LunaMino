import numpy as np
import torch
from .utils import TetrisGame
from .model import TetrisPolicyValue
from .mcts import MCTS
from .reward import get_reward, calculate_heuristics
from . import config

def test_ai_clear_detection(game_seed=42, max_steps=100, mcts_sims=100, device='cpu'):
    """
    测试AI是否看到消行可能性，并是否能看到更高奖励（通过value评估）。

    参数:
    - game_seed: 游戏种子，重现性。
    - max_steps: 最多玩多少步找机会。
    - mcts_sims: MCTS模拟次数。
    - device: 'cpu' 或 'cuda'。

    返回: 字典，结果总结。
    """
    results = {
        'found_opportunity': False,
        'num_clear_moves': 0,
        'chosen_is_clear': False,
        'clear_avg_value': 0.0,
        'non_clear_avg_value': 0.0,
        'message': ''
    }

    # 加载模型
    model = TetrisPolicyValue().to(device)
    try:
        checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        results['message'] = f'加载模型失败: {e}'
        return results

    # 初始化MCTS
    mcts = MCTS(model, device=device, num_simulations=mcts_sims)

    # 初始化游戏
    game = TetrisGame(seed=game_seed)

    step = 0
    while step < max_steps:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            results['message'] = '游戏结束，无动作。'
            break

        # 测试每个move是否消行
        clear_moves = []
        non_clear_moves = []
        for move in legal_moves:
            clone = game.clone()
            res = clone.step(move['x'], move['y'], move['rotation'], move['use_hold'])
            if res['lines_cleared'] > 0:
                clear_moves.append(move)
            else:
                non_clear_moves.append(move)
            del clone

        if clear_moves:
            results['found_opportunity'] = True
            results['num_clear_moves'] = len(clear_moves)

            # 运行MCTS
            root = mcts.run(game)

            # 获取最佳动作（temp=0，贪婪）
            action_probs = mcts.get_action_probs(root, temp=0)
            chosen_global_idx = np.argmax(action_probs)
            if root.legal_indices and chosen_global_idx in root.legal_indices:
                chosen_local_idx = root.legal_indices.index(chosen_global_idx)
                chosen_move = legal_moves[chosen_local_idx]
                results['chosen_is_clear'] = any(chosen_move == cm for cm in clear_moves)

            # 评估value
            clear_values = []
            non_clear_values = []
            sample_non_clear = non_clear_moves[:len(clear_moves)]  # 平衡样本
            for move in clear_moves + sample_non_clear:
                clone = game.clone()
                clone.step(move['x'], move['y'], move['rotation'], move['use_hold'])
                board, ctx = clone.get_state()
                t_board = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                t_ctx = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, value = model(t_board, t_ctx)
                val = value.item()
                if move in clear_moves:
                    clear_values.append(val)
                else:
                    non_clear_values.append(val)
                del clone

            if clear_values:
                results['clear_avg_value'] = np.mean(clear_values)
            if non_clear_values:
                results['non_clear_avg_value'] = np.mean(non_clear_values)

            results['message'] = (
                f'找到{len(clear_moves)}个消行机会。'
                f'AI选择{"是" if results["chosen_is_clear"] else "否"}消行。'
                f'消行后value平均{results["clear_avg_value"]:.3f}，'
                f'非消行{results["non_clear_avg_value"]:.3f}。'
                f'如果消行value不高，网络可能未学好。'
            )
            return results

        # 随机前进
        if legal_moves:
            rand_move = np.random.choice(legal_moves)
            game.step(rand_move['x'], rand_move['y'], rand_move['rotation'], rand_move['use_hold'])
        step += 1

    if not results['found_opportunity']:
        results['message'] = f'{max_steps}步内未找到消行机会，试更大seed或steps。'
    return results

# 示例调用（在脚本末尾运行测试）
if __name__ == "__main__":
    test_result = test_ai_clear_detection(game_seed=42, max_steps=200, mcts_sims=200, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(test_result)