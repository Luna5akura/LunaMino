# ai/mcts.py

import random
from torch import tensor
from torch import zeros
from torch import uint8
from torch import float32
from torch import inference_mode
from torch import amp
import math
import numpy as np
import ctypes
from numba import njit
from .utils import lib, LegalMoves
from .reward import _fast_heuristics
from . import config

@njit(cache=True, fastmath=True, nogil=True)
def _fast_ucb(visits, value_sums, priors, parent_sqrt, c_puct):
    n = len(visits)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if visits[i] > 0:
            q_value = value_sums[i] / visits[i]
        else:
            q_value = 0.0
        u_score = c_puct * priors[i] * parent_sqrt / (1.0 + visits[i])
        scores[i] = q_value + u_score
    return scores

@njit(cache=True, fastmath=True, nogil=True)
def _fast_softmax_normalize(logits, out_probs):
    max_val = -np.inf
    n = len(logits)
    for i in range(n):
        if logits[i] > max_val: max_val = logits[i]
    sum_exp = 0.0
    for i in range(n):
        val = np.exp(logits[i] - max_val)
        out_probs[i] = val
        sum_exp += val
    if sum_exp > 1e-9:
        for i in range(n): out_probs[i] /= sum_exp
    else:
        fill_val = 1.0 / n
        for i in range(n): out_probs[i] = fill_val

class MCTSNode:
    __slots__ = ('stats', 'children', 'num_actions', 'legal_moves', 'action_ids', 'temp_value', 'is_expanded')
    def __init__(self):
        self.num_actions = 0
        self.stats = None
        self.children = None
        self.legal_moves = None
        self.action_ids = None
        self.temp_value = 0.0
        self.is_expanded = False
    
    def get_best_action(self, parent_visits, c_puct):
        parent_sqrt = math.sqrt(max(1, parent_visits))
        scores = _fast_ucb(self.stats[:, 0], self.stats[:, 1], self.stats[:, 2], parent_sqrt, c_puct)
        return np.argmax(scores)

class MCTS:
    def __init__(self, model, device='cuda', num_simulations=50, batch_size=64):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = config.MCTS_C_PUCT
        
        self.c_moves_struct = LegalMoves()
        
        # Pinned Memory buffers
        self.cpu_board_buffer = zeros((batch_size, 200), dtype=uint8, pin_memory=('cuda' in device))
        self.cpu_ctx_buffer = zeros((batch_size, 11), dtype=float32, pin_memory=('cuda' in device))
        
        self.board_ptr_base = self.cpu_board_buffer.data_ptr()
        self.ctx_ptr_base = self.cpu_ctx_buffer.data_ptr()
        self.board_stride = self.cpu_board_buffer.stride(0)
        self.ctx_stride = self.cpu_ctx_buffer.stride(0)

        self._c_get_state = lib.ai_get_state
        self._c_get_legal_moves = lib.ai_get_legal_moves
        self._c_step = lib.ai_step
        self._c_clone = lib.clone_tetris
        self._c_destroy = lib.destroy_tetris

    def _get_legal_moves_fast(self, game_ptr):
        self._c_get_legal_moves(game_ptr, ctypes.byref(self.c_moves_struct))
        count = self.c_moves_struct.count
        if count == 0: return None, None
        
        raw = np.frombuffer(self.c_moves_struct.moves, dtype=np.int8, count=count * 8).reshape(count, 8)
        moves = raw[:, :5]
        ids = raw[:, 6:8].view(np.int16).reshape(-1)
        return moves, ids

    def _evaluate_batch(self, leaf_ptrs, leaf_nodes):
        batch_count = len(leaf_ptrs)
        if batch_count == 0: return

        c_get_state = self._c_get_state
        board_base = self.board_ptr_base
        ctx_base = self.ctx_ptr_base
        board_stride = self.board_stride
        ctx_stride = self.ctx_stride
        
        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        c_float_p = ctypes.POINTER(ctypes.c_float)

        # 1. 填充 CPU Buffer (准备送入 GPU)
        for i in range(batch_count):
            b_addr = board_base + i * board_stride
            c_addr = ctx_base + i * ctx_stride * 4
            c_get_state(leaf_ptrs[i], ctypes.cast(b_addr, c_ubyte_p), ctypes.cast(c_addr, c_float_p))

        # 2. 神经网络推理
        input_board = self.cpu_board_buffer[:batch_count].to(self.device, non_blocking=True).float()
        input_ctx = self.cpu_ctx_buffer[:batch_count].to(self.device, non_blocking=True)
        # 注意：这里保持原有的 view/flip 逻辑
        input_board_gpu = input_board.view(batch_count, 20, 10).flip(1).unsqueeze(1)

        with inference_mode():
             # 如果你的环境必须不用 autocast，这里去掉
            with amp.autocast('cuda' if 'cuda' in self.device else 'cpu'):
                logits_tensor, values_tensor = self.model(input_board_gpu, input_ctx)
        
        logits_np = logits_tensor.float().cpu().numpy()
        values_np = values_tensor.float().cpu().numpy()

        # 3. [新增] 计算启发式分数
        # 直接利用已经在 CPU 内存中的 cpu_board_buffer，避免额外的 GPU->CPU 拷贝
        # cpu_board_buffer 是 (Batch, 200) 的 flat array，我们需要 reshape
        cpu_boards = self.cpu_board_buffer[:batch_count].numpy().reshape(batch_count, 20, 10)
        
        # 启发式权重 (Alpha): 
        # 初期设为 1.0 (全信规则)，随着模型变强可以调低，或者保持 0.5 混合
        # 为了解决你的"冷启动"问题，建议直接给很高，比如 0.8 或 1.0
        # 这里的 value 范围需要归一化到 [-1, 1] 之间以便和 tanh 的输出匹配
        
        for i in range(batch_count):
            node = leaf_nodes[i]
            moves, ids = self._get_legal_moves_fast(leaf_ptrs[i])
            
            if moves is None:
                node.temp_value = -1.0
                continue
            
            node.num_actions = len(moves)
            node.legal_moves = moves
            node.action_ids = ids
            node.children = [None] * node.num_actions
            node.stats = np.zeros((node.num_actions, 3), dtype=np.float32)
            
            raw_logits = logits_np[i][ids]
            _fast_softmax_normalize(raw_logits, node.stats[:, 2])
            
            # --- [修复核心] 混合启发式评分 ---
            nn_val = values_np[i].item()
            

            # 解包所有返回值，最后一个是计算好的 h_val
            max_h, holes, bumpiness, agg_height, h_val = _fast_heuristics(cpu_boards[i])
            
            # 直接使用 ai/reward.py 算出来的 h_val
            node.temp_value = (config.MCTS_VAL_WEIGHT_NN * nn_val + 
                               config.MCTS_VAL_WEIGHT_HEURISTIC * h_val)
            
            node.is_expanded = True

    def run(self, game_state):
        root = MCTSNode()
        root_ptr = self._c_clone(game_state.ptr)
        moves, ids = self._get_legal_moves_fast(root_ptr)
        self._c_destroy(root_ptr)
        
        if moves is None:
            root.temp_value = -1.0
            return root

        board, ctx = game_state.get_state()
        t_b = tensor(board[None, None, ...].copy(), dtype=float32, device=self.device)
        t_c = tensor(ctx[None, ...], dtype=float32, device=self.device)
        
        with inference_mode(), amp.autocast('cuda' if 'cuda' in self.device else 'cpu'):
            logits, _ = self.model(t_b, t_c)
        
        full_logits = logits[0].float().cpu().numpy()
        valid_logits = full_logits[ids]
        probs = np.empty(len(moves), dtype=np.float32)
        _fast_softmax_normalize(valid_logits, probs)
        
        noise = np.random.dirichlet([config.MCTS_DIRICHLET] * len(moves))
        probs = (1 - config.MCTS_EPSILON) * probs + config.MCTS_EPSILON * noise
        
        root.legal_moves = moves
        root.action_ids = ids
        root.num_actions = len(moves)
        root.children = [None] * root.num_actions
        root.stats = np.zeros((root.num_actions, 3), dtype=np.float32)
        root.stats[:, 2] = probs
        root.is_expanded = True

        VIRTUAL_LOSS = 3.0
        leaf_ptrs, leaf_nodes, ptrs_to_free = [], [], []
        path_list = [None] * self.batch_size
        c_step, c_clone = self._c_step, self._c_clone
        
        sim_count = 0
        while sim_count < self.num_simulations:
            current_batch_size = min(self.batch_size, self.num_simulations - sim_count)
            del leaf_ptrs[:]; del leaf_nodes[:]
            valid_indices = []

            for b in range(current_batch_size):
                node = root
                game_ptr = c_clone(game_state.ptr)
                ptrs_to_free.append(game_ptr)
                path = [node]
                
                while node.is_expanded:
                    parent_visits = node.stats[:, 0].sum()
                    idx = node.get_best_action(parent_visits, self.c_puct)
                    node.stats[idx, 0] += 1
                    node.stats[idx, 1] -= VIRTUAL_LOSS
                    path.append(idx)
                    
                    m_x = node.legal_moves[idx, 0]
                    m_y = node.legal_moves[idx, 1]
                    m_r = node.legal_moves[idx, 2]
                    m_h = node.legal_moves[idx, 4]
                    
                    res = c_step(game_ptr, m_x, m_y, m_r, m_h)
                    
                    if res.is_game_over:
                        node = None
                        break
                    
                    child = node.children[idx]
                    if child is None:
                        new_node = MCTSNode()
                        node.children[idx] = new_node
                        node = new_node
                        path.append(node)
                        break
                    else:
                        node = child
                        path.append(node)
                
                path_list[b] = path
                
                if node is not None:
                    leaf_ptrs.append(game_ptr)
                    leaf_nodes.append(node)
                    valid_indices.append(b)
                else:
                    # 关键修复: 追加 None 保持路径结构为 [Node, Action, None]
                    path.append(None)
                    self._backpropagate(path, -1.0, VIRTUAL_LOSS)

            if leaf_ptrs:
                self._evaluate_batch(leaf_ptrs, leaf_nodes)
                for i, b_idx in enumerate(valid_indices):
                    node = leaf_nodes[i]
                    path = path_list[b_idx]
                    self._backpropagate(path, node.temp_value, VIRTUAL_LOSS)

            for ptr in ptrs_to_free: self._c_destroy(ptr)
            del ptrs_to_free[:]
            sim_count += current_batch_size

        if config.DEBUG_MODE and random.random() < 1.0 / config.DEBUG_FREQ:  # 随机采样验证
            parent_visits = root.stats[:, 0].sum()
            best_idx = root.get_best_action(parent_visits, self.c_puct)
            move = root.legal_moves[best_idx]
            x, y, rot, _, hold = move  # 解包动作
            
            valid, prev_b, post_b, res = game_state.validate_step(x, y, rot, hold)
            if not valid:
                print(f"[DEBUG] Inconsistent placement! Action: {move}, Result: {res}")
                # 可选：保存板面为npz文件调试
                np.savez("debug_inconsistent.npz", prev=prev_b, post=post_b)
            else:
                print(f"[DEBUG] Placement consistent for action {move}")
        return root

    def _backpropagate(self, path, value, virtual_loss):
        current_val = value
        for i in range(len(path) - 1, 0, -2): 
            idx = path[i-1]
            node = path[i-2]
            node.stats[idx, 1] += (virtual_loss + current_val)
            current_val *= config.GAMMA

    def get_action_probs(self, root, temp=1.0):
        if root.num_actions == 0: return np.zeros(config.ACTION_DIM, dtype=np.float32)
        visits = root.stats[:, 0]
        
        if np.sum(visits) == 0:
            probs_compact = np.ones(root.num_actions, dtype=np.float32) / root.num_actions
        elif temp < 1e-3:
            probs_compact = np.zeros_like(visits)
            probs_compact[np.argmax(visits)] = 1.0
        else:
            visits_d = visits.astype(np.float64)
            visits_pow = np.power(visits_d, 1.0 / temp)
            sum_visits = np.sum(visits_pow)
            if sum_visits > 1e-9:
                probs_compact = (visits_pow / sum_visits).astype(np.float32)
            else:
                probs_compact = np.ones(root.num_actions, dtype=np.float32) / root.num_actions
           
        full_probs = np.zeros(config.ACTION_DIM, dtype=np.float32)
        full_probs[root.action_ids] = probs_compact
        return full_probs