# ai/utils.py
import ctypes
import os
from . import config
import numpy as np
import platform
from collections import namedtuple

# ==========================================
# 1. 库加载
# ==========================================
def _load_library():
    system = platform.system()
    if system == "Windows":
        lib_name = "libtetris.dll"
    elif system == "Darwin":
        lib_name = "libtetris.dylib"
    else:
        lib_name = "libtetris.so"
    
    # 搜索路径：当前目录 -> ai目录的上一级 -> ai目录
    search_paths = [
        os.path.join(os.getcwd(), lib_name),
        os.path.join(os.path.dirname(__file__), "..", lib_name),
        os.path.join(os.path.dirname(__file__), lib_name),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    
    raise FileNotFoundError(f"Library '{lib_name}' not found inside: {search_paths}")

lib = _load_library()

# ==========================================
# 2. CTypes 结构体
# ==========================================
MAX_LEGAL_MOVES = 256

class StepResultStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("lines_cleared", ctypes.c_int),
        ("damage_sent", ctypes.c_int),
        ("attack_type", ctypes.c_int),
        ("is_game_over", ctypes.c_bool),
        ("b2b_count", ctypes.c_int),
        ("combo_count", ctypes.c_int)
    ]

class MacroAction(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_int8),
        ("y", ctypes.c_int8),
        ("rotation", ctypes.c_int8),
        ("landing_height", ctypes.c_int8),
        ("use_hold", ctypes.c_int8),
        ("padding", ctypes.c_int8),
        ("id", ctypes.c_int16)
    ]

class LegalMoves(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("count", ctypes.c_int),
        ("moves", MacroAction * MAX_LEGAL_MOVES)
    ]

StepInfo = namedtuple('StepInfo', [
    'lines_cleared', 'damage_sent', 'attack_type',
    'is_game_over', 'b2b', 'combo'
])

# ==========================================
# 3. 函数原型绑定
# ==========================================
lib.create_tetris.argtypes = [ctypes.c_int]
lib.create_tetris.restype = ctypes.c_void_p
lib.destroy_tetris.argtypes = [ctypes.c_void_p]
lib.destroy_tetris.restype = None
lib.clone_tetris.argtypes = [ctypes.c_void_p]
lib.clone_tetris.restype = ctypes.c_void_p
lib.ai_reset_game.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ai_get_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_float)
]
lib.ai_get_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(LegalMoves)]
lib.ai_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ai_step.restype = StepResultStruct
lib.ai_receive_garbage.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ai_enable_visualization.argtypes = [ctypes.c_void_p]
lib.ai_render.argtypes = [ctypes.c_void_p]
lib.ai_close_visualization.argtypes = []

# ==========================================
# 4. TetrisGame 类
# ==========================================
class TetrisGame:
    __slots__ = (
        'ptr', '_board_buf', '_board_ptr',
        '_ctx_buf', '_ctx_ptr', '_moves_struct',
        '_np_board_view', '_np_ctx_view',
        '_moves_dtype', '_owns_ptr', '_rendered'
    )

    _c_reset = lib.ai_reset_game
    _c_get_state = lib.ai_get_state
    _c_get_moves = lib.ai_get_legal_moves
    _c_step = lib.ai_step

    def __init__(self, seed=0, ptr=None):
        if ptr:
            self.ptr = ptr
            self._owns_ptr = True
        else:
            self.ptr = lib.create_tetris(seed)
            self._owns_ptr = True
            if not self.ptr:
                raise RuntimeError("Failed to create Tetris instance")
        
        self._rendered = False
        self._board_buf = (ctypes.c_ubyte * 200)()
        self._board_ptr = ctypes.cast(self._board_buf, ctypes.POINTER(ctypes.c_ubyte))
        self._ctx_buf = (ctypes.c_float * 11)()
        self._ctx_ptr = ctypes.cast(self._ctx_buf, ctypes.POINTER(ctypes.c_float))
        self._moves_struct = LegalMoves()
        self._np_board_view = np.ctypeslib.as_array(self._board_buf).reshape(20, 10)
        self._np_ctx_view = np.ctypeslib.as_array(self._ctx_buf)
        self._moves_dtype = np.dtype([
            ('x', 'i1'), ('y', 'i1'), ('rot', 'i1'), ('land', 'i1'), ('hold', 'i1'),
            ('padding', 'i1'), ('id', 'i2')
        ])

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'ptr') and self.ptr:
            if self._rendered:
                try: lib.ai_close_visualization()
                except: pass
            if self._owns_ptr:
                lib.destroy_tetris(self.ptr)
            self.ptr = None

    def clone(self):
        if not self.ptr: raise RuntimeError("Game is closed")
        new_ptr = lib.clone_tetris(self.ptr)
        return TetrisGame(ptr=new_ptr)

    def reset(self, seed=0):
        if not self.ptr: raise RuntimeError("Game is closed")
        lib.ai_reset_game(self.ptr, seed)

    def get_state(self):
        if not self.ptr: raise RuntimeError("Game is closed")
        lib.ai_get_state(self.ptr, self._board_ptr, self._ctx_ptr)

        return self._np_board_view[::-1].copy(), self._np_ctx_view.copy()
        
    def get_legal_moves(self):
        if not self.ptr: raise RuntimeError("Game is closed")
        lib.ai_get_legal_moves(self.ptr, ctypes.byref(self._moves_struct))
        count = self._moves_struct.count
        if count == 0:
            return np.empty((0, 5), dtype=np.int8), np.empty((0,), dtype=np.int64)
                # 1. 获取原始字节视图 (int8)
        raw_bytes = np.frombuffer(self._moves_struct.moves, dtype=np.int8, count=count * 8)
        
        # 2. 重塑为 (N, 8) 矩阵
        matrix = raw_bytes.reshape(count, 8)
        
        # 3. 切片获取前5列 (x, y, rot, land, hold)。这是一个视图，不消耗内存拷贝。
        moves = matrix[:, :5]
        
        # 4. 获取 id。id 在偏移量 6 的位置，长度 2 字节。
        # 我们可以通过视图转换来获取。
        # matrix[:, 6:8] 取出了 id 的字节，然后 .view(np.int16) 将其解释为 short
        # 注意：返回形状会变成 (N, 1)，需要 reshape 扁平化
        ids = matrix[:, 6:8].view(np.int16).reshape(-1)

        if config.DEBUG_MODE:
            print(f"[DEBUG] Legal moves: {moves}")
            print(f"[DEBUG] Legal ids: {ids}")
        
        return moves, ids

    def step(self, x, y, rotation, use_hold):
        # 使用缓存的函数引用
        res = self._c_step(self.ptr, x, y, rotation, use_hold)
        return (res.lines_cleared, res.damage_sent, res.attack_type, 
                res.is_game_over, res.b2b_count, res.combo_count)

    def enable_render(self):
        """显式开启渲染模式"""
        if not self.ptr: return
        lib.ai_enable_visualization(self.ptr)
        self._rendered = True

    def render(self):
        if not self.ptr: return
        if not self._rendered: self.enable_render()
        lib.ai_render(self.ptr)

    def validate_step(self, x, y, rotation, use_hold):
        if not self.ptr: raise RuntimeError("Game is closed")
        # 克隆当前状态
        sim = self.clone()
        prev_board, _ = sim.get_state()
        res = sim.step(x, y, rotation, use_hold)
        post_board, _ = sim.get_state()
        sim.close()
        
        # 简单一致性检查：step后板面应变化，且如果不是game_over，检查是否有新方块放置（e.g., 非空行增加）
        changed = not np.array_equal(prev_board, post_board)
        valid = changed and (res[3] == False or res[0] > 0)  # 示例规则：变化且非意外结束
        
        return valid, prev_board, post_board, res