# ai/utils.py

import ctypes
import os
import numpy as np
import weakref

# ==========================================
# 1. 库加载与配置
# ==========================================
LIB_PATH = os.path.join(os.path.dirname(__file__), "../libtetris.so")
if not os.path.exists(LIB_PATH):
    # 尝试当前目录，方便调试
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libtetris.so")
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError(f"Shared library not found at: {LIB_PATH}")

lib = ctypes.CDLL(LIB_PATH)

# ==========================================
# 2. CTypes 结构体定义
# ==========================================
MAX_LEGAL_MOVES = 256

class MacroAction(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int8),
        ("y", ctypes.c_int8),
        ("rotation", ctypes.c_int8),
        ("landing_height", ctypes.c_int8),
        ("use_hold", ctypes.c_bool)
    ]

class LegalMoves(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("moves", MacroAction * MAX_LEGAL_MOVES)
    ]

class StepResult(ctypes.Structure):
    _fields_ = [
        ("lines_cleared", ctypes.c_int),
        ("damage_sent", ctypes.c_int),
        ("attack_type", ctypes.c_int),
        ("is_game_over", ctypes.c_bool),
        ("b2b_count", ctypes.c_int),
        ("combo_count", ctypes.c_int)
    ]

# ==========================================
# 3. 函数原型定义
# ==========================================
lib.create_tetris.argtypes = [ctypes.c_int]
lib.create_tetris.restype = ctypes.c_void_p

lib.destroy_tetris.argtypes = [ctypes.c_void_p]
lib.destroy_tetris.restype = None

lib.clone_tetris.argtypes = [ctypes.c_void_p]
lib.clone_tetris.restype = ctypes.c_void_p

lib.ai_reset_game.argtypes = [ctypes.c_void_p, ctypes.c_int]

# Pointers for buffers
lib.ai_get_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),  # board (200)
    ctypes.POINTER(ctypes.c_int),  # queue (5)
    ctypes.POINTER(ctypes.c_int),  # hold (1)
    ctypes.POINTER(ctypes.c_int)   # meta (5)
]

lib.ai_get_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(LegalMoves)]

lib.ai_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ai_step.restype = StepResult

lib.ai_receive_garbage.argtypes = [ctypes.c_void_p, ctypes.c_int]

lib.ai_enable_visualization.argtypes = [ctypes.c_void_p]
lib.ai_render.argtypes = [ctypes.c_void_p]
lib.ai_close_visualization.argtypes = []


# ==========================================
# 4. 高性能 Python 封装
# ==========================================
class TetrisGame:

    __slots__ = (

        '__weakref__',  # <--- 必须添加这一项才能使用 weakref.finalize
        'ptr', '_rendered', '_board_buf', '_board_ptr', 
        '_queue_buf', '_queue_ptr', '_hold_buf', '_hold_ptr', 
        '_meta_buf', '_meta_ptr', '_moves_struct', '_context_buf',
        'np_board', 'np_queue', 'np_hold', 'np_meta', 
        '_moves_np_view', '_struct_size'
    )

    def __init__(self, seed=0, ptr=None):
        if ptr:
            self.ptr = ptr
        else:
            self.ptr = lib.create_tetris(seed)
            if not self.ptr:
                raise RuntimeError("Failed to create Tetris instance")
        
        self._rendered = False
        
        # --- Buffer Allocation ---
        # Board: 10x20 = 200 ints
        self._board_buf = (ctypes.c_int * 200)()
        self._board_ptr = ctypes.cast(self._board_buf, ctypes.POINTER(ctypes.c_int))
        
        # Queue: 5 ints
        self._queue_buf = (ctypes.c_int * 5)()
        self._queue_ptr = ctypes.cast(self._queue_buf, ctypes.POINTER(ctypes.c_int))
        
        # Hold: 1 int
        self._hold_buf = (ctypes.c_int * 1)()
        self._hold_ptr = ctypes.cast(self._hold_buf, ctypes.POINTER(ctypes.c_int))
        
        # Meta: 5 ints
        self._meta_buf = (ctypes.c_int * 5)()
        self._meta_ptr = ctypes.cast(self._meta_buf, ctypes.POINTER(ctypes.c_int))
        
        # Legal Moves Struct
        self._moves_struct = LegalMoves()
        
        # Context numpy buffer (pre-allocated)
        self._context_buf = np.zeros(11, dtype=np.int32)
        
        # --- Zero-copy Views (Standard) ---
        # 注意：这里 [::-1] 创建了一个负步长的视图，这在 PyTorch 转换时通常需要 copy()
        self.np_board = np.ctypeslib.as_array(self._board_buf).reshape(20, 10)[::-1]
        
        # 使用 int32 视图，方便快速索引
        self.np_queue = np.ctypeslib.as_array(self._queue_buf)
        self.np_hold = np.ctypeslib.as_array(self._hold_buf)
        self.np_meta = np.ctypeslib.as_array(self._meta_buf)
        
        # --- Optimization: Persistent Moves View ---
        # 计算结构体实际大小（包含 padding）
        self._struct_size = ctypes.sizeof(MacroAction)
        # 直接基于 buffer 创建一个大的 numpy 视图，避免每次调用 create_buffer
        # shape: (256, struct_size_in_bytes) -> 视作 int8 字节流
        self._moves_np_view = np.frombuffer(self._moves_struct.moves, dtype=np.int8).reshape(MAX_LEGAL_MOVES, self._struct_size)
        
        weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        if self.ptr:
            if self._rendered:
                try:
                    lib.ai_close_visualization()
                except:
                    pass
                self._rendered = False
            
            try:
                lib.destroy_tetris(self.ptr)
            except:
                pass
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def clone(self):
        new_ptr = lib.clone_tetris(self.ptr)
        # 注意：这里会重新分配 Python 侧的 Buffer，这是必须的，因为每个 Game 实例需要独立的 Buffer
        return TetrisGame(ptr=new_ptr)

    def reset(self, seed=0):
        lib.ai_reset_game(self.ptr, seed)

    def get_state(self):
        """
        获取游戏状态。
        :return: (board_view, context_array)
        警告: board_view 是底层 C 内存的视图。如果需要保存历史状态，调用者必须进行 .copy()。
        """
        lib.ai_get_state(
            self.ptr,
            self._board_ptr,
            self._queue_ptr,
            self._hold_ptr,
            self._meta_ptr
        )
        
        # Optimization: 使用 NumPy 数组索引而非 Ctypes 属性访问
        # Ctypes access (e.g. self._meta_buf[3]) creates a Python int object.
        # Numpy access (e.g. self.np_meta[3]) reads directly from memory (faster).
        
        # Context Mapping:
        # 0: current_piece (meta[3])
        # 1: hold (hold[0])
        # 2-6: queue[0-4]
        # 7: b2b (meta[0])
        # 8: combo (meta[1])
        # 9: can_hold (meta[2])
        # 10: pending_garbage (meta[4])
        
        self._context_buf[0] = self.np_meta[3]
        self._context_buf[1] = self.np_hold[0]
        self._context_buf[2:7] = self.np_queue[:]
        self._context_buf[7:10] = self.np_meta[:3]
        self._context_buf[10] = self.np_meta[4]
        
        return self.np_board, self._context_buf

    def get_legal_moves(self):
        """
        :return: Copy of legal moves array (N, 5).
        """
        lib.ai_get_legal_moves(self.ptr, ctypes.byref(self._moves_struct))
        
        count = self._moves_struct.count
        if count == 0:
            return np.empty((0, 5), dtype=np.int8)
            
        # Optimization: 直接切片预创建的 numpy 视图，然后 copy
        # 必须 copy，因为 _moves_struct 内存会在下一次 get_legal_moves 时被覆盖
        # 取前 5 个字节 (x, y, rot, land, hold)
        return self._moves_np_view[:count, :5].copy()

    def step(self, x, y, rotation, use_hold):
        """
        :return: (lines, sent, attack_type, game_over, combo)
        """
        # ctypes call overhead is dominant here, hard to optimize further without C changes
        res = lib.ai_step(self.ptr, x, y, rotation, use_hold)
        
        return (
            res.lines_cleared,
            res.damage_sent,
            res.attack_type,
            res.is_game_over, # bool
            res.combo_count
        )

    def receive_garbage(self, lines):
        lib.ai_receive_garbage(self.ptr, lines)

    def enable_render(self):
        lib.ai_enable_visualization(self.ptr)
        self._rendered = True

    def render(self):
        if self._rendered:
            lib.ai_render(self.ptr)