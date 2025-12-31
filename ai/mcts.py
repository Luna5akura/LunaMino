# ai/mcts.py

import math
import numpy as np
import torch
from numba import njit
from torch.amp import autocast

# 引入优化后的 utils
from .utils import TetrisGame, lib
from . import config

# ==========================================
# 1. Numba 加速的 UCB 计算
# ==========================================
@njit(cache=True, fastmath=True)
def _fast_ucb(visits, value_sums, priors, parent_sqrt, c_puct):
    """
    使用 Numba 编译 UCB 计算，避免 NumPy 的临时数组创建和 Python 解释器开销。
    直接在原生 C 数组上操作。
    """
    n = len(visits)
    scores = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        # Q(s, a)
        if visits[i] > 0:
            q_value = value_sums[i] / visits[i]
        else:
            q_value = 0.0
            
        # U(s, a)
        u_score = c_puct * priors[i] * parent_sqrt / (1.0 + visits[i])
        
        scores[i] = q_value + u_score
        
    return scores

# ==========================================
# 2. 优化后的 Node 结构
# ==========================================
class MCTSNode:
    __slots__ = ('legal_moves', 'stats', 'children', 'num_actions', 'is_expanded')

    def __init__(self, prior_probs, legal_moves):
        self.legal_moves = legal_moves
        self.num_actions = len(legal_moves)
        
        # stats layout: [N, 3] -> (visit_count, value_sum, prior)
        # float32 足够且更省内存
        self.stats = np.zeros((self.num_actions, 3), dtype=np.float32)
        self.stats[:, 2] = prior_probs
        
        self.children = [None] * self.num_actions
        self.is_expanded = True

    def get_best_action(self, parent_visits, c_puct, virtual_loss=0.0):
        # 计算 sqrt 放到外部或这里均可
        parent_sqrt = math.sqrt(max(1, parent_visits))
        
        # 提取 Numba 需要的列 (Numpy 切片是视图，开销很小)
        visits = self.stats[:, 0]
        value_sums = self.stats[:, 1]
        priors = self.stats[:, 2]
        
        # 调用 JIT 函数
        scores = _fast_ucb(visits, value_sums, priors, parent_sqrt, c_puct)
        
        # 应用 Virtual Loss 惩罚（如果启用）
        # 如果某个节点正在被 Batch 中的其他线程访问，它的分数会暂时降低
        if virtual_loss > 0:
            # 这里的逻辑略复杂：Virtual Loss 通常加在 visit_count 和 value_sum 上
            # 为简单起见，我们在外部控制 stats 的临时修改，或者在这里直接计算
            # 标准做法是在 traverse 时修改 stats，backprop 时还原
            pass

        return np.argmax(scores)

# ==========================================
# 3. 优化后的 MCTS 类
# ==========================================
class MCTS:
    def __init__(self, model, device='cuda', num_simulations=50, batch_size=16):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = config.MCTS_C_PUCT
        
        # 预分配 CUDA Tensor 缓冲区，避免每次 malloc
        # 假设最大 batch_size
        # Shape: (B, 1, 20, 10) 匹配优化后的 model 输入
        self.tensor_board_buf = torch.zeros((batch_size, 1, 20, 10), dtype=torch.float32, device=device)
        self.tensor_ctx_buf = torch.zeros((batch_size, 11), dtype=torch.float32, device=device)

        # 预分配 CPU Numpy 缓冲区，用于收集 Batch 数据
        self.np_board_buf = np.zeros((batch_size, 1, 20, 10), dtype=np.float32)
        self.np_ctx_buf = np.zeros((batch_size, 11), dtype=np.float32)

    def _evaluate_batch_optimized(self, games, legal_move_lists):
        """
        极速批量评估：
        1. 使用预分配的 CPU Buffer 收集数据
        2. 一次性转换为 Tensor 并传输到 GPU
        """
        valid_indices = []
        count = 0
        
        for i, game in enumerate(games):
            if game is not None:
                board, ctx = game.get_state() # Returns views
                
                # 填充 CPU Buffer (注意 copy)
                # board 是 (20, 10)，需要 reshape 到 (1, 20, 10)
                self.np_board_buf[count, 0] = board 
                self.np_ctx_buf[count] = ctx
                
                valid_indices.append(i)
                count += 1
        
        if count == 0:
            return None, None, []

        # 核心优化：一次性传输
        # 使用切片只传输有效部分
        t_boards = torch.from_numpy(self.np_board_buf[:count]).to(self.device, non_blocking=True)
        t_ctxs = torch.from_numpy(self.np_ctx_buf[:count]).to(self.device, non_blocking=True)

        # Inference
        with torch.no_grad(), autocast(device_type='cuda'):
            logits, values = self.model(t_boards, t_ctxs)
        
        # GPU -> CPU (Batch transfer)
        logits_np = logits.float().cpu().numpy()
        values_np = values.float().cpu().numpy()
        
        # 后处理
        probs_list = []
        for i in range(count):
            list_idx = valid_indices[i]
            num_legal = len(legal_move_lists[list_idx])
            
            # Extract valid logits
            valid_logits = logits_np[i, :num_legal]
            
            # Numerically stable Softmax
            max_l = np.max(valid_logits)
            exps = np.exp(valid_logits - max_l)
            probs = exps / np.sum(exps)
            probs_list.append(probs)
            
        return probs_list, values_np, valid_indices

    def run(self, game_state: TetrisGame):
        """
        执行 MCTS 搜索。
        """
        # --- 1. Root Expansion ---
        legal = game_state.get_legal_moves()
        if len(legal) == 0:
            return MCTSNode(np.array([], dtype=np.float32), np.array([]))

        # 快速评估根节点
        board, ctx = game_state.get_state()
        
        # Single instance evaluation
        t_board = torch.tensor(board.copy(), dtype=torch.float32, device=self.device).view(1, 1, 20, 10)
        t_ctx = torch.tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad(), autocast(device_type='cuda'):
            logits, _ = self.model(t_board, t_ctx)
        
        logits = logits[0].float().cpu().numpy()
        valid_logits = logits[:len(legal)]
        
        # Softmax & Noise
        probs = np.exp(valid_logits - np.max(valid_logits))
        probs /= np.sum(probs)
        
        noise = np.random.dirichlet([config.MCTS_DIRICHLET] * len(legal))
        probs = (1 - config.MCTS_EPSILON) * probs + config.MCTS_EPSILON * noise
        
        root = MCTSNode(probs, legal)
        # 初始访问计数设为1，避免 sqrt(0)
        # root.stats[:, 0] += 1e-8 

        # --- 2. Simulation Loop (Batched) ---
        # 虚拟损失常量
        VIRTUAL_LOSS = 3.0 
        
        for _ in range(0, self.num_simulations, self.batch_size):
            # 动态调整 Batch 大小
            current_batch_size = min(self.batch_size, self.num_simulations - _)
            
            paths = []
            leaf_games = []
            leaf_legal_moves = []
            raw_ptrs_to_free = []
            
            # --- Selection Phase (Parallel-ish) ---
            for _ in range(current_batch_size):
                node = root
                game_ptr = lib.clone_tetris(game_state.ptr)
                raw_ptrs_to_free.append(game_ptr)
                
                path = [node]
                
                while True:
                    if node.num_actions == 0:
                        break
                    
                    # 计算父节点访问量 (Sum of visits to children)
                    parent_visits = np.sum(node.stats[:, 0])
                    
                    # Numba UCB
                    action_idx = node.get_best_action(parent_visits, self.c_puct)
                    
                    # === 关键优化：Virtual Loss ===
                    # 立即增加访问计数和扣除 Value，防止 Batch 中后续的 Simulation 走完全一样的路
                    node.stats[action_idx, 0] += 1        # Virtual Visit
                    node.stats[action_idx, 1] -= VIRTUAL_LOSS # Virtual Value Penalty
                    
                    path.append(action_idx)
                    
                    # 获取 Move 数据
                    move = node.legal_moves[action_idx]
                    
                    # 在 C 副本上执行一步
                    res = lib.ai_step(game_ptr, move[0], move[1], move[2], move[4])
                    
                    if res.is_game_over:
                        node = None
                        break
                    
                    child = node.children[action_idx]
                    if child is None:
                        # Found leaf
                        node = None 
                        break
                    else:
                        node = child
                        path.append(node)
                
                # Wrap leaf game for evaluation
                # 注意：如果游戏结束，leaf_games 对应位置填 None
                if node is None and len(path) > 0 and isinstance(path[-1], int):
                    # 把指针所有权移交给 TetrisGame 对象
                    leaf_game = TetrisGame(ptr=game_ptr)
                    raw_ptrs_to_free.pop() # 不再需要在 finally 块释放，由 leaf_game 析构
                    
                    moves = leaf_game.get_legal_moves()
                    if len(moves) == 0:
                        # Terminal state logic
                        leaf_games.append(None)
                        leaf_legal_moves.append([])
                        # 游戏结束判负
                        self._backpropagate(path, -1.0, root, VIRTUAL_LOSS) 
                    else:
                        leaf_games.append(leaf_game)
                        leaf_legal_moves.append(moves)
                        paths.append(path)
                else:
                    # 这种情况通常是 node.num_actions == 0 或其他边缘情况
                    # 需要回退 Virtual Loss
                    self._revert_virtual_loss(path, root, VIRTUAL_LOSS)

            # --- Evaluation Phase (GPU Batch) ---
            if leaf_games:
                probs_list, values_np, valid_indices = self._evaluate_batch_optimized(leaf_games, leaf_legal_moves)
                
                if probs_list:
                    # --- Expansion & Backpropagation ---
                    for i, list_idx in enumerate(valid_indices):
                        probs = probs_list[i]
                        value = values_np[i].item()
                        moves = leaf_legal_moves[list_idx]
                        path = paths[i] # 这里的 path 与 valid_indices 对应关系要小心
                        
                        # paths 列表包含所有 batch 的路径，需要通过 indices 映射吗？
                        # 不需要，valid_indices 是 leaf_games 的索引，而 leaf_games 和 paths 是同步 append 的。
                        # wait: leaf_games.append 在循环里，如果中间有 None (GameOver)，indices 会跳过。
                        # 修正：Evaluate batch 返回的 values_np 长度等于 valid_indices 长度
                        # valid_indices[i] 指的是原始 leaf_games 列表中的下标 j
                        # 而 paths[j] 就是对应的路径。
                        
                        original_idx = list_idx
                        current_path = paths[original_idx]
                        
                        parent_node = current_path[-2]
                        action_idx = current_path[-1]
                        
                        # Expand
                        new_node = MCTSNode(probs, moves)
                        parent_node.children[action_idx] = new_node
                        
                        # Backpropagate (Real Value) + Revert Virtual Loss
                        self._backpropagate(current_path, value, root, VIRTUAL_LOSS)
            
            # Cleanup leftover pointers
            for ptr in raw_ptrs_to_free:
                lib.destroy_tetris(ptr)
            
            # Cleanup game objects
            # (leaf_games hold pointers, they will be freed when list is cleared/functions exit)
            pass

        return root

    def _revert_virtual_loss(self, path, root, virtual_loss):
        """如果在 Selection 阶段中断（如无路可走），需要回退 Virtual Loss"""
        for i in range(len(path) - 1, 0, -2):
            action_idx = path[i]
            node = path[i-1]
            node.stats[action_idx, 0] -= 1
            node.stats[action_idx, 1] += virtual_loss

    def _backpropagate(self, path, value, root, virtual_loss):
        """
        1. 还原 Virtual Loss (Visit-1, Value+VL)
        2. 应用真实更新 (Visit+1, Value+RealVal)
        简化后：Visit 不变(因为 VL 已经 +1 了，真实也是 +1)，Value = Value + VL + RealVal
        """
        current_val = value
        # Path: [Node, Action, Node, Action, ...]
        # Traverse backwards
        for i in range(len(path) - 1, 0, -2):
            action_idx = path[i]
            node = path[i-1]
            
            # Revert Virtual Loss
            # node.stats[action_idx, 0] -= 1  <-- 不需要减，因为真实访问也要 +1
            node.stats[action_idx, 1] += virtual_loss
            
            # Apply Real Update
            # node.stats[action_idx, 0] += 1 <-- 不需要加，VL 阶段已经加了
            node.stats[action_idx, 1] += current_val
            
            # 注意：如果要在多线程/异步环境中使用，这里需要原子操作或锁
            # 但我们在单线程中运行 Batch，所以是安全的。

    def get_action_probs(self, root, temp=1.0):
        """
        获取动作概率，并填充到固定长度 (256)，以便存入 ReplayBuffer
        """
        visits = root.stats[:, 0]
        
        # 计算当前合法动作的概率
        if temp == 0:
            best_idx = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
        else:
            visits_temp = visits ** (1.0 / temp)
            total = np.sum(visits_temp)
            if total > 0:
                probs = visits_temp / total
            else:
                probs = np.ones_like(visits) / len(visits)
        
        # === 修复开始：填充到固定维度 (256) ===
        # 必须与 model.py 中的 action_dim 和 train.py 中的 buffer 保持一致
        ACTION_DIM = 256 
        full_probs = np.zeros(ACTION_DIM, dtype=np.float32)
        
        # 将计算出的概率填入对应位置
        # 注意：这里假设网络输出的前 N 个节点对应 N 个合法动作
        num_legal = len(probs)
        if num_legal > 0:
            # 防止溢出 (理论上不应该发生，除非 legal_moves 超过 256)
            limit = min(num_legal, ACTION_DIM)
            full_probs[:limit] = probs[:limit]
            
        return full_probs
        # === 修复结束 ===