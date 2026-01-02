# ai/inference.py

import torch
import numpy as np
import time
import os
from multiprocessing import shared_memory
from . import config

# 状态标志
FLAG_IDLE = 0
FLAG_REQ_READY = 1
FLAG_RES_READY = 2

class InferenceBatcher:
    """
    管理共享内存和批处理逻辑的核心类。
    实现原理：
    1. 开辟一块巨大的共享内存，划分为 N 个 Slot (N = Worker数量)。
    2. Worker i 只能读写 Slot i。
    3. Server 扫描所有 Slot，将状态为 REQ_READY 的打包送入 GPU，
       算完后写回并标记为 RES_READY。
    """
    def __init__(self, num_workers, action_dim=config.ACTION_DIM):
        self.num_workers = num_workers
        self.action_dim = action_dim
        
        # 预计算各部分的大小 (Bytes)
        # Board: [N, 20, 10] (int8)
        self.size_board = num_workers * 200 
        # Ctx: [N, 11] (float32 -> 4 bytes)
        self.size_ctx = num_workers * 11 * 4
        # Logits: [N, 2304] (float16 -> 2 bytes, 节省带宽)
        self.size_logits = num_workers * action_dim * 2
        # Value: [N, 1] (float32 -> 4 bytes)
        self.size_value = num_workers * 1 * 4
        # Flags: [N] (uint8)
        self.size_flags = num_workers * 1
        
        self.total_size = (self.size_board + self.size_ctx + 
                           self.size_logits + self.size_value + self.size_flags)
        
        # 计算偏移量
        self.offset_board = 0
        self.offset_ctx = self.offset_board + self.size_board
        self.offset_logits = self.offset_ctx + self.size_ctx
        self.offset_value = self.offset_logits + self.size_logits
        self.offset_flags = self.offset_value + self.size_value

        # 创建共享内存
        # 注意：在主进程创建，Worker 连接
        self.shm = None
        self.shm_name = None
        
        # Numpy 视图 (用于读写)
        self.np_boards = None
        self.np_ctxs = None
        self.np_logits = None
        self.np_values = None
        self.np_flags = None

    def create(self):
        """仅在主进程调用：分配内存"""
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)
            self.shm_name = self.shm.name
            self._init_arrays()
            # 初始化 flag 为 0
            self.np_flags[:] = FLAG_IDLE
            print(f"[InferenceBatcher] Shared Memory created: {self.shm_name} ({self.total_size/1024/1024:.2f} MB)")
            return self.shm_name
        except FileExistsError:
            # 防止上次未清理干净
            print("[Error] Shared memory already exists. Please cleanup manually or restart.")
            raise

    def connect(self, shm_name):
        """在 Worker 进程调用：连接内存"""
        self.shm_name = shm_name
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self._init_arrays()

    def close(self):
        if self.shm:
            self.shm.close()
            
    def unlink(self):
        """仅在主进程退出时调用"""
        if self.shm:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    def _init_arrays(self):
        """基于 Buffer 创建 Numpy 视图 (Zero-Copy)"""
        buf = self.shm.buf
        
        self.np_boards = np.ndarray((self.num_workers, 20, 10), dtype=np.int8, buffer=buf, offset=self.offset_board)
        self.np_ctxs = np.ndarray((self.num_workers, 11), dtype=np.float32, buffer=buf, offset=self.offset_ctx)
        # Logits 使用 float16 传输，减少带宽压力
        self.np_logits = np.ndarray((self.num_workers, self.action_dim), dtype=np.float16, buffer=buf, offset=self.offset_logits)
        self.np_values = np.ndarray((self.num_workers, 1), dtype=np.float32, buffer=buf, offset=self.offset_value)
        self.np_flags = np.ndarray((self.num_workers,), dtype=np.uint8, buffer=buf, offset=self.offset_flags)

# =========================================================================
# Client Side: 伪装成 Model，供 MCTS 调用
# =========================================================================
class RemoteModel:
    def __init__(self, batcher: InferenceBatcher, worker_idx: int):
        self.batcher = batcher
        self.idx = worker_idx
        
    def __call__(self, board_tensor, ctx_tensor):
        """
        模拟 torch.nn.Module 的调用接口。
        输入是 Tensor，但我们需要转 numpy 写入共享内存。
        """
        # 1. Tensor -> Numpy (假设输入是 CPU tensor)
        # board_tensor: [1, 1, 20, 10], ctx_tensor: [1, 11]
        # squeeze 掉 batch 维度
        b_np = board_tensor[0, 0].numpy() # int8 or uint8
        c_np = ctx_tensor[0].numpy()
        
        # 2. 写入共享内存 (Zero-copy write)
        self.batcher.np_boards[self.idx] = b_np
        self.batcher.np_ctxs[self.idx] = c_np
        
        # 3. 设置标志位：告诉 Server 我准备好了
        self.batcher.np_flags[self.idx] = FLAG_REQ_READY
        
        # 4. 自旋等待 (Spin-wait)
        # 对于 MCTS 这种高频低延迟场景，sleep 开销太大，自旋虽然占 CPU 但延迟最低
        # 为防止死锁，加个简单的 timeout 保护（虽然理论上不需要）
        # 在 Python 中纯 pass 循环会抢占 GIL，适当加极其微小的 sleep 甚至更好
        while self.batcher.np_flags[self.idx] != FLAG_RES_READY:
             # time.sleep(0.000001) # 微秒级 sleep，让出 GIL 
             pass
        
        # 5. 读取结果
        # 拷贝出来，因为共享内存马上要被覆写
        logits = self.batcher.np_logits[self.idx].astype(np.float32)
        value = self.batcher.np_values[self.idx] # float32
        
        # 6. 重置标志位
        self.batcher.np_flags[self.idx] = FLAG_IDLE
        
        # 7. 转回 Tensor 格式返回给 MCTS
        # MCTS 期望: logits [1, 2304], values [1, 1]
        t_logits = torch.from_numpy(logits).unsqueeze(0)
        t_values = torch.from_numpy(value).unsqueeze(0)
        
        return t_logits, t_values

    def eval(self): pass # 兼容接口
    def to(self, device): return self # 兼容接口

# =========================================================================
# Server Side: 跑在主进程或独立线程
# =========================================================================
def run_inference_loop(batcher: InferenceBatcher, model, device, stop_event):
    """
    独立线程函数：不断扫描共享内存，批量推理
    """
    print("[Inference Server] Started.")
    
    # 预分配 GPU 缓冲区 (Pinned Memory)
    # 这样从共享内存 copy 到 Tensor 会更快
    pin_boards = torch.zeros((batcher.num_workers, 1, 20, 10), dtype=torch.uint8).pin_memory()
    pin_ctxs = torch.zeros((batcher.num_workers, 11), dtype=torch.float32).pin_memory()
    
    while not stop_event.is_set():
        # 1. 扫描标志位 (numpy bool indexing 很快)
        # 找到所有状态为 REQ_READY (1) 的 worker 索引
        ready_mask = (batcher.np_flags == FLAG_REQ_READY)
        indices = np.where(ready_mask)[0]
        
        if len(indices) == 0:
            # 没有请求，稍微睡一下避免烧 CPU
            time.sleep(0.0001) 
            continue
            
        # 2. 收集数据 (Batching)
        # 直接从共享内存切片读取
        batch_boards = torch.as_tensor(batcher.np_boards[indices]) # int8
        batch_ctxs = torch.as_tensor(batcher.np_ctxs[indices])
        
        # 3. 数据上 GPU
        # 注意：这里我们做了一个 copy 到 pinned memory，然后再 to(device)
        # 对于小 batch，直接 to(device) 可能也够快，视情况而定
        
        # 调整形状 [B, 20, 10] -> [B, 1, 20, 10]
        current_b = batch_boards.unsqueeze(1).to(device, non_blocking=True).float()
        current_c = batch_ctxs.to(device, non_blocking=True)
        
        # 4. 模型推理
        # 强制 float32 以兼容 NixOS 环境
        with torch.inference_mode():
            logits, values = model(current_b, current_c)
            
        # 5. 写回结果
        # Tensor(GPU) -> Numpy(CPU Shared Memory)
        # logits: [B, 2304], values: [B, 1]
        
        # 转回 float16 写入共享内存
        out_logits = logits.cpu().numpy().astype(np.float16)
        out_values = values.cpu().numpy()
        
        batcher.np_logits[indices] = out_logits
        batcher.np_values[indices] = out_values
        
        # 6. 通知 Workers
        batcher.np_flags[indices] = FLAG_RES_READY
        
        # 统计 (可选)
        # print(f"Batched {len(indices)} reqs")