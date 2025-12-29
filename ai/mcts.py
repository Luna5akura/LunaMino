# ai/mcts.py (adjusted for new model structure)
import math
import numpy as np
import torch
from .utils import TetrisGame
from . import config # 引入 config 以用 MCTS_C_PUCT
class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {} # Map relative_idx (0 to len-1) -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior # P(s, a) from neural net
        self.is_expanded = False
        self.legal_moves = None # 新增: 存储当前状态的 legal_moves
        self.legal_indices = None # 新增: 存储合法动作的全局索引 (0-79)
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    def ucb_score(self, c_puct=config.MCTS_C_PUCT): # 从 config 取，增强探索
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
            noise = np.random.dirichlet([config.MCTS_DIRICHLET] * legal_count)
            epsilon = config.MCTS_EPSILON
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
                leaf_value = -1.0 # Penalty for dying
           
            # 4. Backup
            final_val = leaf_value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += final_val # 单人游戏，不翻转
       
        return root
    def _expand_node(self, node, game):
        if node.is_expanded:
            return 0
       
        board, ctx = game.get_state()
       
        # 获取合法动作
        legal_moves = game.get_legal_moves()
        node.legal_moves = legal_moves # 存储在节点
       
        num_legal = len(legal_moves)
        if num_legal == 0:
            node.is_expanded = True
            return 0
       
        # 计算 logits 和 value
        t_board = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
       
        with torch.no_grad():
            logits, value = self.model(t_board, t_ctx)
       
        logits = logits[0] # [80]
       
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
       
        node.legal_indices = legal_indices # 新增: 保存 legal_indices
       
        node.is_expanded = True
        return value.item()

    def get_action_probs(self, root, temp=1.0):
        legal_count = len(root.legal_moves) if root.legal_moves else 0
        counts = np.zeros(legal_count)
        for i in range(legal_count):
            if i in root.children:
                counts[i] = root.children[i].visit_count
    
        if temp == 0:
            if legal_count == 0:
                probs = np.zeros(legal_count)
            else:
                max_count = np.max(counts)
                best_idxs = np.flatnonzero(counts == max_count) # 处理 ties
                if len(best_idxs) == 0: # 全 0，fallback uniform
                    probs = np.ones(legal_count) / legal_count
                else:
                    best_idx = np.random.choice(best_idxs)
                    probs = np.zeros(legal_count)
                    probs[best_idx] = 1.0
        else:
            counts = counts ** (1.0 / temp)
            sum_counts = np.sum(counts)
            if sum_counts > 0:
                probs = counts / sum_counts
            else: # 全 0，fallback uniform
                probs = np.ones(legal_count) / legal_count if legal_count > 0 else np.zeros(legal_count)
    
        # 强制归一化，确保 sum=1 (处理浮点精度或遗漏情况)
        if legal_count > 0:
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs /= sum_p # 修正轻微偏差
            else:
                probs = np.ones(legal_count) / legal_count # 强制 uniform
    
        # 投影到固定 80 维向量，非法动作概率=0
        full_probs = np.zeros(80)
        if root.legal_indices is not None:
            for i in range(legal_count):
                idx = root.legal_indices[i]
                full_probs[idx] = probs[i]
    
        # 始终规范化 full_probs 以确保 sum == 1.0 精确
        sum_full = np.sum(full_probs)
        has_nan = np.any(np.isnan(full_probs))
        if has_nan:
            full_probs[np.isnan(full_probs)] = 0
            sum_full = np.sum(full_probs)
        if sum_full == 0 and legal_count > 0:
            uniform_p = 1.0 / legal_count
            full_probs = np.zeros(80)
            for i in range(legal_count):
                idx = root.legal_indices[i]
                full_probs[idx] = uniform_p
            sum_full = np.sum(full_probs)
        if sum_full > 0:
            full_probs /= sum_full
    
        # 新增: 检查无效 probs (NaN 或 sum !=1)，如果仍有问题，打印但已处理
        sum_full_after = np.sum(full_probs)
        has_nan_after = np.any(np.isnan(full_probs))
        if has_nan_after or not np.isclose(sum_full_after, 1.0, rtol=1e-8):
            print(f"[DEBUG] Post-normalization invalid! sum={sum_full_after}, has_nan={has_nan_after}, legal_count={legal_count}.")
    
        return full_probs # (80,)