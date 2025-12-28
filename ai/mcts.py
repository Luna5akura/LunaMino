# ai/mcts.py

import math
import numpy as np
import torch
from .utils import TetrisGame
from . import config  # 引入 config 以用 MCTS_C_PUCT

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  # Map relative_idx (0 to len-1) -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  # P(s, a) from neural net
        self.is_expanded = False
        self.legal_moves = None  # 新增: 存储当前状态的 legal_moves

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=config.MCTS_C_PUCT):  # 从 config 取，增强探索
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + u

class MCTS:
    def __init__(self, model, device='cuda', num_simulations=50):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.model.eval()

    def run(self, game_state: TetrisGame):
        root = MCTSNode()
        
        # 1. Expand root immediately
        self._expand_node(root, game_state)
        legal_count = len(root.children)
        if legal_count > 0:
            noise = np.random.dirichlet([1.0] * legal_count)
            epsilon = 0.5  # 25% 噪声
            for i in range(legal_count):
                child = root.children[i]
                child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
        
        for _ in range(self.num_simulations):
            node = root
            sim_game = game_state.clone()
            
            # 2. Selection (Traverse down the tree)
            path = [node]
            while node.is_expanded and len(node.children) > 0:
                # Select child with highest UCB
                action_idx, node = max(node.children.items(), key=lambda item: item[1].ucb_score())
                path.append(node)
                
                # 新: 从 parent (path[-2]) 的 legal_moves 取 move
                parent = path[-2]
                move = parent.legal_moves[action_idx]
                
                # 执行 step (现在支持 y)
                res = sim_game.step(move['x'], move['y'], move['rotation'], move['use_hold'])
                
                if res['game_over']:
                    break
            
            # 3. Expansion & Evaluation
            leaf_value = 0
            if not res['game_over']:
                leaf_value = self._expand_node(node, sim_game)
            else:
                leaf_value = -1.0  # Penalty for dying
            
            # 4. Backup
            final_val = leaf_value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += final_val  # 单人游戏，不翻转
        
        return root

    def _expand_node(self, node, game):
        if node.is_expanded:
            return 0
        
        board, ctx, p_type = game.get_state()
        
        # 获取合法动作
        legal_moves = game.get_legal_moves()
        node.legal_moves = legal_moves  # 存储在节点
        
        num_legal = len(legal_moves)
        if num_legal == 0:
            node.is_expanded = True
            return 0
        
        # 计算 logits 和 value
        t_board = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ptype = torch.tensor([p_type], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits, value = self.model(t_board, t_ctx, t_ptype)
        
        logits = logits[0]  # [80]
        
        # Masking: 只对合法 idx 取 logit
        mask = torch.full_like(logits, -float('inf'))
        legal_indices = []
        for move in legal_moves:
            base_idx = move['x'] * 4 + move['rotation']
            idx = base_idx + 40 if move['use_hold'] else base_idx
            legal_indices.append(idx)
            mask[idx] = logits[idx]
        
        probs_t = torch.softmax(mask, dim=0).cpu().numpy()
        
        # 提取 probs 只为 legal_moves
        probs = np.zeros(num_legal)
        for i, idx in enumerate(legal_indices):
            probs[i] = probs_t[idx]
        
        # 创建 children，用 0 to num_legal-1 作为 key
        node.children = {}
        for i in range(num_legal):
            child = MCTSNode(parent=node, prior=probs[i])
            node.children[i] = child
        
        node.is_expanded = True
        return value.item()

    def get_action_probs(self, root, temp=1.0):
        legal_count = len(root.legal_moves) if root.legal_moves else 0
        counts = np.zeros(legal_count)
        for i in range(legal_count):
            if i in root.children:
                counts[i] = root.children[i].visit_count
        
        if temp == 0:
            best_idx = np.argmax(counts)
            probs = np.zeros(legal_count)
            probs[best_idx] = 1.0
            return probs
        
        counts = counts ** (1.0 / temp)
        sum_counts = np.sum(counts)
        if sum_counts > 0:
            probs = counts / sum_counts
        else:
            probs = np.zeros(legal_count)
        
        return probs