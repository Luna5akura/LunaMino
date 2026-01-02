# ai/mcts.py

import math
import numpy as np
import torch
import ctypes
from numba import njit
from .utils import lib, LegalMoves
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
        self.cpu_board_buffer = torch.zeros((batch_size, 200), dtype=torch.uint8, pin_memory=('cuda' in device))
        self.cpu_ctx_buffer = torch.zeros((batch_size, 11), dtype=torch.float32, pin_memory=('cuda' in device))
        
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

        for i in range(batch_count):
            b_addr = board_base + i * board_stride
            c_addr = ctx_base + i * ctx_stride * 4
            c_get_state(leaf_ptrs[i], ctypes.cast(b_addr, c_ubyte_p), ctypes.cast(c_addr, c_float_p))

        input_board = self.cpu_board_buffer[:batch_count].to(self.device, non_blocking=True).float()
        input_ctx = self.cpu_ctx_buffer[:batch_count].to(self.device, non_blocking=True)
        input_board = input_board.view(batch_count, 20, 10).flip(1).unsqueeze(1)

        with torch.inference_mode():
            with torch.amp.autocast('cuda' if 'cuda' in self.device else 'cpu'):
                logits_tensor, values_tensor = self.model(input_board, input_ctx)
        
        logits_np = logits_tensor.float().cpu().numpy()
        values_np = values_tensor.float().cpu().numpy()

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
            node.temp_value = values_np[i].item()
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
        t_b = torch.tensor(board[None, None, ...].copy(), dtype=torch.float32, device=self.device)
        t_c = torch.tensor(ctx[None, ...], dtype=torch.float32, device=self.device)
        
        with torch.inference_mode(), torch.amp.autocast('cuda' if 'cuda' in self.device else 'cpu'):
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