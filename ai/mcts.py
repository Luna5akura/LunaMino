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
            damage_reward = min(res.get('damage_sent', 0) / 4.0, 1.0)
            final_val = leaf_value + damage_reward
            
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += final_val
                # 这里暂时不翻转 value，假设单人最大化分数
                # final_val = -final_val 

        return root

    def _expand_node(self, node, game):
        if node.is_expanded:
            return 0
            
        board, ctx, p_type = game.get_state()
        
        # To Tensor
        t_board = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ptype = torch.tensor([p_type], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits, value = self.model(t_board, t_ctx, t_ptype)
            
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # [80]
        
        # Get Legal Moves from C
        legal_moves = game.get_legal_moves() 
        
        node.children = {}
        valid_probs_sum = 0
        
        for move in legal_moves:
            # --- [修复] 编码动作索引 (LegalMoves -> 0-79) ---
            base_idx = move['x'] * 4 + move['rotation']
            if move['use_hold']:
                idx = 40 + base_idx
            else:
                idx = base_idx
                
            if 0 <= idx < 80:
                p = probs[idx]
                child = MCTSNode(parent=node, prior=p)
                node.children[idx] = child
                valid_probs_sum += p
                
        # Re-normalize priors
        if valid_probs_sum > 0:
            for child in node.children.values():
                child.prior /= valid_probs_sum
                
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