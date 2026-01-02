# ai/buffer.py

import numpy as np
import os
from . import config

class NumpyReplayBuffer:
    def __init__(self, capacity, board_shape=(20, 10), ctx_dim=11, action_dim=config.ACTION_DIM):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 保持 int8 以节省内存 (200 bytes per state)
        self.boards = np.zeros((capacity, 1, *board_shape), dtype=np.int8)
        self.ctxs = np.zeros((capacity, ctx_dim), dtype=np.float32)
        # float16 足够存储概率，节省一半内存
        self.probs = np.zeros((capacity, action_dim), dtype=np.float16)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

    def add_batch(self, boards, ctxs, probs, values):
        """
        优化: 使用切片写入代替模运算索引，大幅提升大 Batch 写入速度。
        """
        # 确保输入是 numpy 数组 (避免在切片赋值时隐式转换)
        # 注意: 这里的 np.array 可能会产生一次拷贝，如果外部能直接传 numpy 进来最好
        if not isinstance(boards, np.ndarray): boards = np.array(boards, dtype=np.int8)
        if not isinstance(ctxs, np.ndarray): ctxs = np.array(ctxs, dtype=np.float32)
        if not isinstance(probs, np.ndarray): probs = np.array(probs, dtype=np.float16)
        if not isinstance(values, np.ndarray): values = np.array(values, dtype=np.float32)

        n = len(boards)
        if n == 0: return

        # 调整 shape 匹配 (N, 1, H, W)
        if boards.ndim == 3:
            boards = boards[:, np.newaxis, :, :]
        
        values = values.reshape(-1, 1)

        # 逻辑: 判断是否需要回绕
        space_left = self.capacity - self.ptr
        
        if n <= space_left:
            # Case 1: 剩余空间足够，直接切片写入 (最快路径)
            end = self.ptr + n
            self.boards[self.ptr:end] = boards
            self.ctxs[self.ptr:end] = ctxs
            self.probs[self.ptr:end] = probs
            self.values[self.ptr:end] = values
            self.ptr = (self.ptr + n) % self.capacity
        else:
            # Case 2: 需要回绕，分两段写入
            # 第一段: 填满 buffer 尾部
            self.boards[self.ptr:] = boards[:space_left]
            self.ctxs[self.ptr:] = ctxs[:space_left]
            self.probs[self.ptr:] = probs[:space_left]
            self.values[self.ptr:] = values[:space_left]
            
            # 第二段: 剩余的写到 buffer 头部
            remainder = n - space_left
            self.boards[:remainder] = boards[space_left:]
            self.ctxs[:remainder] = ctxs[space_left:]
            self.probs[:remainder] = probs[space_left:]
            self.values[:remainder] = values[space_left:]
            
            self.ptr = remainder

        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """
        优化: 移除 astype(float32)，直接返回原始 dtype。
        数据传输到 GPU 后再转 float，减少 PCI-E 带宽压力并利用 GPU 加速转换。
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.boards[idxs], # 返回 int8
            self.ctxs[idxs],   # 返回 float32
            self.probs[idxs],  # 返回 float16
            self.values[idxs]  # 返回 float32
        )

    def save(self, filepath, ui):
        ui.log(f"Saving buffer to {filepath}...")
        try:
            # 优化: 使用 savez_compressed 代替 pickle
            # 压缩率更高，且专为 numpy 数组设计
            np.savez_compressed(
                filepath,
                ptr=np.array(self.ptr),
                size=np.array(self.size),
                boards=self.boards[:self.size],
                ctxs=self.ctxs[:self.size],
                probs=self.probs[:self.size],
                values=self.values[:self.size]
            )
        except Exception as e:
            ui.log(f"[red]Buffer save failed: {e}[/red]")

    def load(self, filepath, ui):
        if not os.path.exists(filepath):
            ui.log(f"[yellow]Buffer file {filepath} not found, starting fresh.[/yellow]")
            return

        ui.log(f"Loading buffer from {filepath}...")
        try:
            with np.load(filepath) as data:
                self.ptr = int(data['ptr'])
                self.size = int(data['size'])
                
                # 直接切片赋值
                self.boards[:self.size] = data['boards']
                self.ctxs[:self.size] = data['ctxs']
                self.probs[:self.size] = data['probs']
                self.values[:self.size] = data['values']
                
            ui.log(f"Loaded {self.size} samples.")
        except Exception as e:
            ui.log(f"[yellow]Buffer load failed/reset: {e}[/yellow]")
            # 出错时重置，防止 ptr 越界
            self.ptr = 0
            self.size = 0