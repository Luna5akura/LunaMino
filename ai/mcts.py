# ai/mcts.py
import math
import numpy as np
import torch
from .utils import TetrisGame
from . import config

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
        self.legal_moves = None
        # 移除 legal_indices（不再需要）

    @property
    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def ucb_score(self, parent_sqrt, c_puct=config.MCTS_C_PUCT):
        # UCB = Q + U
        u = c_puct * self.prior * parent_sqrt / (1 + self.visit_count)
        return self.value + u

class MCTS:
    def __init__(self, model, device='cuda', num_simulations=50, batch_size=8):
        self.model = model.to(device)
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.model.eval()

    def _expand_node(self, node, game):
        """
        扩展节点：获取合法动作，并使用神经网络预测先验概率(Prior)。
        """
        board, ctx = game.get_state()
      
        # 获取合法动作 (Numpy array: [N, 5])
        node.legal_moves = game.get_legal_moves()
        num_legal = len(node.legal_moves)
        if num_legal == 0:
            node.is_expanded = True
            return
      
        # 核心修复: board 来自 [::-1] 切片，存在负步长，PyTorch 不支持。
        # 必须先 .copy() 变为连续内存。
        t_board = torch.tensor(board.copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(t_board, t_ctx)
      
        logits = logits[0]  # Remove batch dim
      
        # Masking: 只保留前 num_legal 个动作的概率，其他设为 -inf
        mask = torch.full_like(logits, -float('inf'))  # logits 是 256 维
        mask[:num_legal] = logits[:num_legal]  # 只激活前 num_legal 个
      
        # 计算 Softmax
        probs_t = torch.softmax(mask, dim=0)
        probs = probs_t.cpu().numpy()  # 只取前 num_legal 个 (其余 prob=0)
      
        # 初始化子节点 (key 为 0 到 num_legal-1)
        node.children = {j: MCTSNode(parent=node, prior=probs[j]) for j in range(num_legal)}
        node.is_expanded = True

    def run(self, game_state: TetrisGame):
        root = MCTSNode()
        self._expand_node(root, game_state)
      
        # 添加 Dirichlet 噪声到根节点以鼓励探索
        legal_count = len(root.children)
        if legal_count > 0:
            noise = np.random.dirichlet([config.MCTS_DIRICHLET] * legal_count)
            epsilon = config.MCTS_EPSILON
            for i in range(legal_count):
                root.children[i].prior = (1 - epsilon) * root.children[i].prior + epsilon * noise[i]
      
        # 模拟循环
        for sim in range(0, self.num_simulations, self.batch_size):
            batch_size = min(self.batch_size, self.num_simulations - sim)
          
            leaves = []
            paths = []
            sim_games = []
            # 1. Selection & Simulation (到达叶子节点)
            for _ in range(batch_size):
                node = root
                sim_game = game_state.clone()  # Clone C++ state
                path = [node]
              
                # 贪婪选择直到遇到未扩展节点或游戏结束
                while node.is_expanded and len(node.children) > 0:
                    parent_sqrt = math.sqrt(node.visit_count)
                    # Select best child by UCB
                    action_idx = max(node.children, key=lambda i: node.children[i].ucb_score(parent_sqrt))
                  
                    node = node.children[action_idx]
                    path.append(node)
                  
                    # 在模拟环境中执行动作
                    parent = path[-2]
                    move = parent.legal_moves[action_idx]  # action_idx 即 local_idx (0..num_legal-1)
                    # move: [x, y, rot, land, hold]
                    sim_game.step(move[0], move[1], move[2], move[4])
                paths.append(path)
                leaves.append(node)
                sim_games.append(sim_game)
          
            # 2. Evaluation (神经网络推断)
            if leaves:
                boards = []
                ctxs = []
                terminals = [False] * len(leaves)
                values = [0.0] * len(leaves)
                # 收集叶子节点状态
                for i, (leaf, sim_game) in enumerate(zip(leaves, sim_games)):
                    # 如果已经在 Selection 阶段判定为终局（之前已扩展但无子节点）
                    if leaf.is_expanded:
                        if len(leaf.children) == 0:
                            terminals[i] = True
                            values[i] = -1.0  # Loss
                    else:
                        # 尚未扩展，获取合法动作检查是否结束
                        leaf.legal_moves = sim_game.get_legal_moves()
                        num_legal = len(leaf.legal_moves)
                        if num_legal == 0:
                            terminals[i] = True
                            values[i] = -1.0
                            leaf.is_expanded = True  # 标记为扩展（死局）
                        else:
                            # 准备推断数据
                            board, ctx = sim_game.get_state()
                            # 修复: 负步长拷贝
                            boards.append(torch.tensor(board.copy(), dtype=torch.float32, device=self.device))
                            ctxs.append(torch.tensor(ctx, dtype=torch.float32, device=self.device))
              
                # 批量推断
                logits_batch = None
                values_batch = None
                if boards:
                    t_boards = torch.stack(boards)
                    t_ctxs = torch.stack(ctxs)
                    with torch.no_grad():
                        logits_batch, values_batch = self.model(t_boards, t_ctxs)
              
                # 3. Expansion & Backpropagation
                batch_idx = 0
                for i, (leaf, sim_game) in enumerate(zip(leaves, sim_games)):
                    if terminals[i]:
                        leaf_value = values[i]
                    else:
                        # 使用网络输出扩展节点
                        logits = logits_batch[batch_idx]
                        value = values_batch[batch_idx].item()
                      
                        num_legal = len(leaf.legal_moves)
                        mask = torch.full_like(logits, -float('inf'))  # 256 维
                        mask[:num_legal] = logits[:num_legal]
                        probs_t = torch.softmax(mask, dim=0)
                        probs = probs_t[:num_legal].cpu().numpy()
                        leaf.children = {j: MCTSNode(parent=leaf, prior=probs[j]) for j in range(num_legal)}
                        leaf.is_expanded = True
                      
                        leaf_value = value
                        batch_idx += 1
                  
                    # 反向传播
                    # 注意：通常 Value Network 输出是对当前局面的估值 (-1输, 1赢)
                    # 如果当前是 P1 的回合，这个 Value 是对 P1 有利的程度。
                    # MCTS 路径上的节点交替? 单人 Tetris 不需要 Negamax 翻转，直接加和即可。
                    final_val = leaf_value
                    for n in reversed(paths[i]):
                        n.visit_count += 1
                        n.value_sum += final_val
          
            # 及时释放 C++ 内存，防止内存泄漏
            for g in sim_games:
                g.close()
      
        return root

    def get_action_probs(self, root, temp=1.0):
        """
        根据根节点的访问次数计算动作概率分布
        """
        legal_count = len(root.legal_moves) if root.legal_moves is not None else 0
        counts = torch.zeros(256)  # 修改：固定256维
      
        for i in range(legal_count):
            if i in root.children:
                counts[i] = root.children[i].visit_count
      
        if temp == 0:
            # 贪婪模式 (Argmax)
            if legal_count == 0:
                probs = torch.zeros(256)
            else:
                max_count = torch.max(counts[:legal_count])  # 只看前legal_count
                best_idxs = torch.nonzero(counts[:legal_count] == max_count).flatten()
                if len(best_idxs) == 0:
                    probs = torch.ones(256) / 256
                else:
                    best_idx = best_idxs[torch.randint(0, len(best_idxs), (1,))]
                    probs = torch.zeros(256)
                    probs[best_idx] = 1.0
        else:
            # 温度采样 (整个counts，非法=0)
            counts = counts ** (1.0 / temp)
            sum_counts = torch.sum(counts)
            if sum_counts > 0:
                probs = counts / sum_counts
            else:
                probs = torch.ones(256) / 256

        # 归一化 (已确保 sum>0 或 uniform)
        if probs.sum() > 0:
            probs /= probs.sum()

        return probs.numpy()  # 固定256维