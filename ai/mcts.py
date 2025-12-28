# ai/mcts.py
import math
import numpy as np
import torch
from .utils import TetrisGame

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {} # Map move_index -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior # P(s, a) from neural net
        self.is_expanded = False
    
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.0):
        # 避免除以0
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
            noise = np.random.dirichlet([0.3] * legal_count)
            epsilon = 0.25 # 25% 的决策由噪声决定
            for i, (idx, child) in enumerate(root.children.items()):
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
                
                # --- [修复] 解码动作索引 (0-79) ---
                use_hold = 0
                decoded_idx = action_idx
                if decoded_idx >= 40:
                    use_hold = 1
                    decoded_idx -= 40
                
                x = decoded_idx // 4
                rot = decoded_idx % 4
                
                # --- [修复] 传递 use_hold 参数 ---
                res = sim_game.step(x, rot, use_hold)
                
                if res['game_over']:
                    break
            
            # 3. Expansion & Evaluation
            leaf_value = 0
            if not res['game_over']:
                 # Use Neural Net to evaluate leaf
                leaf_value = self._expand_node(node, sim_game)
            else:
                leaf_value = -1.0 # Penalty for dying
            
            # 4. Backup

            final_val = leaf_value 
            
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += final_val
                # final_val = -final_val # 单人游戏不需要取反
            # damage_reward = min(res.get('damage_sent', 0) / 4.0, 1.0)
            # final_val = leaf_value + damage_reward
            
            # for n in reversed(path):
            #     n.visit_count += 1
            #     n.value_sum += final_val
            #     # 这里暂时不翻转 value，假设单人最大化分数
            #     # final_val = -final_val 

        return root
    def _expand_node(self, node, game):
        if node.is_expanded:
            return 0
            
        board, ctx, p_type = game.get_state()
        
        # 1. 获取所有合法动作的索引列表
        legal_moves = game.get_legal_moves()
        # print(f'{len(legal_moves)=}')
        legal_indices = []
        for move in legal_moves:
            base_idx = move['x'] * 4 + move['rotation']
            if move['use_hold']:
                idx = 40 + base_idx
            else:
                idx = base_idx
            if 0 <= idx < 80:
                legal_indices.append(idx)
        
        # To Tensor
        t_board = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ptype = torch.tensor([p_type], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits, value = self.model(t_board, t_ctx, t_ptype)
            
        logits = logits[0] # [80]
        
        # --- [关键修改] Action Masking ---
        # 创建一个全是负无穷的 mask
        mask = torch.full_like(logits, -float('inf'))
        # 只在合法位置填入原来的 logit
        if len(legal_indices) > 0:
            mask[legal_indices] = logits[legal_indices]
        else:
            # 极罕见情况：无路可走（死局），但也得防 crash
            pass
            
        # 对 mask 后的 logits 做 softmax
        # 这样非法动作的概率直接变为 0，合法动作的概率和为 1
        probs = torch.softmax(mask, dim=0).cpu().numpy() 
        
        # --- 修改结束 ---

        node.children = {}
        # 不需要 valid_probs_sum 了，因为 softmax 已经归一化了
        
        for idx in legal_indices:
            p = probs[idx]
            child = MCTSNode(parent=node, prior=p)
            node.children[idx] = child
                
        node.is_expanded = True
        return value.item()
    def get_action_probs(self, root, temp=1.0):
        # Returns vector size 80
        counts = np.zeros(80) # [修复] 大小改为 80
        for idx, child in root.children.items():
            counts[idx] = child.visit_count
            
        if temp == 0:
            best_idx = np.argmax(counts)
            probs = np.zeros(80)
            probs[best_idx] = 1.0
            return probs
            
        counts = counts ** (1.0 / temp)
        sum_counts = np.sum(counts)
        if sum_counts > 0:
            probs = counts / sum_counts
        else:
            probs = np.zeros(80) # 防止除以0
            
        return probs