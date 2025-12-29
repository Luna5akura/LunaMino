# ai/utils.py (with modifications for debugging and context inclusion)
import ctypes
import os
import random
import numpy as np
# 加载动态库
LIB_PATH = os.path.join(os.path.dirname(__file__), "../libtetris.so")
lib = ctypes.CDLL(LIB_PATH)
# ==========================================
# C 结构体定义 (必须与 tetris.h 保持一致)
# ==========================================
class MacroAction(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("rotation", ctypes.c_int),
        ("landing_height", ctypes.c_int),
        ("use_hold", ctypes.c_int) # 新增
    ]
class LegalMoves(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("moves", MacroAction * 200) # MAX_LEGAL_MOVES
    ]
class StepResult(ctypes.Structure):
    _fields_ = [
        ("lines_cleared", ctypes.c_int),
        ("damage_sent", ctypes.c_int),
        ("attack_type", ctypes.c_int),
        ("is_game_over", ctypes.c_int), # 确保这里是 c_int
        ("b2b_count", ctypes.c_int),
        ("combo_count", ctypes.c_int)
    ]
# 定义函数签名
lib.create_tetris.argtypes = [ctypes.c_int]
lib.create_tetris.restype = ctypes.c_void_p
lib.destroy_tetris.argtypes = [ctypes.c_void_p]
lib.clone_tetris.argtypes = [ctypes.c_void_p]
lib.clone_tetris.restype = ctypes.c_void_p
lib.ai_receive_garbage.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ai_reset_game.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ai_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib.ai_get_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(LegalMoves)]
lib.ai_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] # ptr, x, y, rotation, use_hold
lib.ai_step.restype = StepResult
lib.ai_receive_garbage.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ai_enable_visualization.argtypes = [ctypes.c_void_p]
lib.ai_render.argtypes = [ctypes.c_void_p]
lib.ai_close_visualization.argtypes = []
# ==========================================
# Python Wrapper
# ==========================================
class TetrisGame:
    def __init__(self, seed=None, ptr=None):
        if ptr:
            self.ptr = ptr
            self.owned = True
        else:
            if seed is None:
                seed = 0 # Let C use time(NULL) for auto different
            self.ptr = lib.create_tetris(seed)
            if not self.ptr:
                raise RuntimeError("Failed to create Tetris")
            self.owned = True
    def __del__(self):
        if self.owned and self.ptr:
            lib.destroy_tetris(self.ptr)
            self.ptr = None
    def clone(self):
        new_ptr = lib.clone_tetris(self.ptr)
        return TetrisGame(ptr=new_ptr)
    def get_state(self):
        """
        返回神经网络需要的 Tensor 数据
        Feature Map: 10x20 Board + Context info
        """
        board_buf = (ctypes.c_int * 200)()
        queue_buf = (ctypes.c_int * 5)()
        hold_buf = (ctypes.c_int * 1)()
        meta_buf = (ctypes.c_int * 4)() # b2b, ren, can_hold, current_piece_type
        lib.ai_get_state(self.ptr, board_buf, queue_buf, hold_buf, meta_buf)
        # 构造 Feature Planes (Channels, Height, Width)
        # Channel 0: Board (0/1)
        # Channel 1: Current Piece Indicator (One-hot ideally, but here simplified)
        # Channel 2: Ghost Piece (Optional, skipped for now)
      
        board = np.array(board_buf).reshape(20, 10) # y, x
        board = np.flip(board, axis=0).copy()
      
        # 翻转 y 轴，让底部在下面 (如果在 C 中 y=0 是底部)
        # 通常神经网络喜欢 y=0 在左上角或符合直觉，这里保持 C 的原始数据
        # C 代码: y=0 是底部。
      
        # 构造 Context Vector
        # [Current Type, Hold Type, Next1, Next2, Next3, Next4, Next5, B2B, Combo, CanHold]
        current_piece_type = meta_buf[3]
        context = np.concatenate([
            np.array([current_piece_type]),
            np.array(hold_buf),
            np.array(queue_buf),
            np.array(meta_buf[:3]) # b2b, ren, can_hold
        ])
      
        return board, context
    def get_legal_moves(self):
        """返回所有合法落点 [(x, y, rot), ...]"""
        moves_struct = LegalMoves()
        lib.ai_get_legal_moves(self.ptr, ctypes.byref(moves_struct))
      
        results = []
        for i in range(moves_struct.count):
            m = moves_struct.moves[i]
            results.append({
                'x': m.x,
                'y': m.y,
                'rotation': m.rotation,
                'use_hold': m.use_hold,
                'landing_height': m.landing_height # Not fully implemented in C yet, but struct has it
            })
        # print(f"[DEBUG] Legal moves count: {len(results)}") # 新: 调试 print
        return results
    def step(self, x, y, rotation, use_hold):
        res = lib.ai_step(self.ptr, x, y, rotation, use_hold)
        return {
            'lines_cleared': res.lines_cleared,
            'damage_sent': res.damage_sent,
            'attack_type': res.attack_type,
            'game_over': bool(res.is_game_over),
            'combo': res.combo_count
        }
    def receive_garbage(self, lines):
        lib.ai_receive_garbage(self.ptr, lines)
    def enable_render(self):
        """开启 Raylib 窗口"""
        lib.ai_enable_visualization(self.ptr)
    def render(self):
        """绘制一帧"""
        lib.ai_render(self.ptr)
  
    def close_render(self):
        lib.ai_close_visualization()
