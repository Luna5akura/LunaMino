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
    raise FileNotFoundError(f"Shared library not found at: {LIB_PATH}")

lib = ctypes.CDLL(LIB_PATH)

# ==========================================
# 2. CTypes 结构体定义 (严格匹配 C 头文件)
# ==========================================
MAX_LEGAL_MOVES = 256  # 必须匹配 bridge.h

class MacroAction(ctypes.Structure):
    # 【修复】移除 _pack_ = 1，让 ctypes 自动匹配 C 编译器的默认对齐
    # _pack_ = 1  <-- 删除这行
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
        ("is_game_over", ctypes.c_bool), # 修正：必须对应 C 的 bool (1 byte)
        # C 编译器通常会在这里自动插入 3 bytes padding 以对齐下面的 int
        ("b2b_count", ctypes.c_int),
        ("combo_count", ctypes.c_int)
    ]

# ==========================================
# 3. 函数原型定义 (明确类型以防 Segfault)
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
    ctypes.POINTER(ctypes.c_int), # board (200)
    ctypes.POINTER(ctypes.c_int), # queue (5)
    ctypes.POINTER(ctypes.c_int), # hold (1)
    ctypes.POINTER(ctypes.c_int)  # meta (5) - 修正大小
]

lib.ai_get_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(LegalMoves)]

lib.ai_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ai_step.restype = StepResult

lib.ai_receive_garbage.argtypes = [ctypes.c_void_p, ctypes.c_int]

# Visualization
lib.ai_enable_visualization.argtypes = [ctypes.c_void_p]
lib.ai_render.argtypes = [ctypes.c_void_p]
lib.ai_close_visualization.argtypes = []

# ==========================================
# 4. 高性能 Python 封装
# ==========================================
class TetrisGame:
    def __init__(self, seed=0, ptr=None):
        """
        :param seed: 随机种子
        :param ptr: 如果不为None，接管现有的C指针（用于clone）
        """
        if ptr:
            self.ptr = ptr
        else:
            self.ptr = lib.create_tetris(seed)
            if not self.ptr:
                raise RuntimeError("Failed to create Tetris instance")
        
        # 资源管理标志
        self._rendered = False
        
        # --- 性能优化：预分配内存缓冲区 ---
        # Board: 10x20 = 200 ints
        self._board_buf = (ctypes.c_int * 200)()
        self._board_ptr = ctypes.cast(self._board_buf, ctypes.POINTER(ctypes.c_int))
        
        # Queue: 5 ints
        self._queue_buf = (ctypes.c_int * 5)()
        self._queue_ptr = ctypes.cast(self._queue_buf, ctypes.POINTER(ctypes.c_int))
        
        # Hold: 1 int
        self._hold_buf = (ctypes.c_int * 1)()
        self._hold_ptr = ctypes.cast(self._hold_buf, ctypes.POINTER(ctypes.c_int))
        
        # Meta: 5 ints (b2b, combo, can_hold, cur_piece, pending_garbage)
        self._meta_buf = (ctypes.c_int * 5)()
        self._meta_ptr = ctypes.cast(self._meta_buf, ctypes.POINTER(ctypes.c_int))
        
        # Legal Moves 结构体复用
        self._moves_struct = LegalMoves()

        # --- 性能优化：Numpy 零拷贝视图 (Zero-copy Views) ---
        # 这些数组直接指向 ctypes 缓冲区的内存地址。
        # 当调用 C 函数填充 buffer 后，这些 numpy 数组会自动“看到”新数据，无需复制。
        self.np_board = np.ctypeslib.as_array(self._board_buf).reshape(20, 10)[::-1]
        
        self.np_queue = np.ctypeslib.as_array(self._queue_buf)
        self.np_hold  = np.ctypeslib.as_array(self._hold_buf)
        self.np_meta  = np.ctypeslib.as_array(self._meta_buf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # 安全的析构函数：不引用全局 lib，防止解释器退出时报错
        self.close()

    def close(self):
        if hasattr(self, 'ptr') and self.ptr:
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

    def clone(self):
        """创建当前游戏状态的副本"""
        new_ptr = lib.clone_tetris(self.ptr)
        return TetrisGame(ptr=new_ptr)

    def reset(self, seed=0):
        """重置游戏"""
        lib.ai_reset_game(self.ptr, seed)

    def get_state(self):
        """
        获取游戏状态。
        :return: (board_array, meta_array)
        """
        # 1. 调用 C 填充缓冲区
        lib.ai_get_state(
            self.ptr, 
            self._board_ptr, 
            self._queue_ptr, 
            self._hold_ptr, 
            self._meta_ptr
        )
        
        # 2. 组装 Context
        # Meta Indices: 0:b2b, 1:combo, 2:can_hold, 3:cur_piece, 4:pending_garbage
        
        current_piece = self._meta_buf[3]
        
        # --- 核心修复：确保拼接后的长度为 11 ---
        # 之前的版本只有 10 个 (1+1+5+3)
        # 现在加上 pending_garbage (1) -> 总共 11
        context = np.concatenate([
            [current_piece],       # 1
            self.np_hold,          # 1
            self.np_queue,         # 5
            self.np_meta[:3],      # 3 (b2b, combo, can_hold)
            [self.np_meta[4]]      # 1 (pending_garbage) <--- 这一项必须加上
        ])
        
        # 此时 len(context) 应为 11
        
        return self.np_board, context

    def get_legal_moves(self):
        """
        获取合法动作。
        :return: Numpy 数组 shape (N, 5) -> [x, y, rotation, landing_height, use_hold]
        """
        lib.ai_get_legal_moves(self.ptr, ctypes.byref(self._moves_struct))
        
        count = self._moves_struct.count
        # print(f'\n\n COUNT {count=}\n\n')
        if count == 0:
            return np.empty((0, 5), dtype=np.int8)

        # 【修复】使用 ctypes 迭代读取，确保内存对齐正确
        # 虽然比 np.frombuffer 慢一点点，但能避免读取到 struct padding 垃圾值
        moves = self._moves_struct.moves
        result = np.zeros((count, 5), dtype=np.int8)
        
        for i in range(count):
            m = moves[i]
            result[i, 0] = m.x
            result[i, 1] = m.y
            result[i, 2] = m.rotation
            result[i, 3] = m.landing_height
            result[i, 4] = int(m.use_hold)
        
        return result

    def step(self, x, y, rotation, use_hold):
        """
        执行一步。
        :return: (lines, sent, game_over, combo) 元组，比 dict 快
        """

        res = lib.ai_step(self.ptr, x, y, rotation, use_hold)
        
        return (
            res.lines_cleared,
            res.damage_sent,
            res.attack_type,
            bool(res.is_game_over),
            res.combo_count
        )

    def receive_garbage(self, lines):
        lib.ai_receive_garbage(self.ptr, lines)

    # --- Visualization ---
    def enable_render(self):
        lib.ai_enable_visualization(self.ptr)
        self._rendered = True

    def render(self):
        if self._rendered:
            lib.ai_render(self.ptr)