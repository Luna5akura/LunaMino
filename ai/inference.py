# ai/inference.py

from torch import inference_mode
from torch import as_tensor
from torch import from_numpy
from torch import zeros
from torch import uint8
from torch import float32
import numpy as np
import time
import os
from multiprocessing import shared_memory
from . import config

class InferenceBatcher:
    """
    管理共享内存和批处理逻辑的核心类。
    实现原理：
    1. 开辟一块巨大的共享内存，划分为 N 个 Slot (N = Worker数量)。
    2. Worker i 只能读写 Slot i。
    3. Server 扫描所有 Slot，将状态为 REQ_READY 的打包送入 GPU，
       算完后写回并标记为 RES_READY。
    """
    def __init__(self, num_workers, local_batch_size=config.MCTS_BATCH_SIZE, action_dim=config.ACTION_DIM):
        self.num_workers = num_workers
        self.local_batch_size = local_batch_size # 新增维度
        self.action_dim = action_dim
        
        # 计算大小：现在每个 Worker 拥有 local_batch_size 个槽位
        # Shape 变为: [num_workers, local_batch_size, ...]
        total_slots = num_workers * local_batch_size
        
        self.size_board = total_slots * 200 
        self.size_ctx = total_slots * 11 * 4
        self.size_logits = total_slots * action_dim * 2
        self.size_value = total_slots * 1 * 4
        # Flag 依然是每个 Worker 一个，因为是一次性提交一批
        self.size_flags = num_workers * 1 
        
        self.total_size = (self.size_board + self.size_ctx + 
                           self.size_logits + self.size_value + self.size_flags)
        
        # ... (Offset 计算逻辑不变，只是 base size 变大了) ...
        self.offset_board = 0
        self.offset_ctx = self.offset_board + self.size_board
        self.offset_logits = self.offset_ctx + self.size_ctx
        self.offset_value = self.offset_logits + self.size_logits
        self.offset_flags = self.offset_value + self.size_value

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
            self.np_flags[:] = config.FLAG_IDLE
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
        buf = self.shm.buf
        N = self.num_workers
        B = self.local_batch_size
        
        # 增加维度 B
        self.np_boards = np.ndarray((N, B, 20, 10), dtype=np.int8, buffer=buf, offset=self.offset_board)
        self.np_ctxs = np.ndarray((N, B, 11), dtype=np.float32, buffer=buf, offset=self.offset_ctx)
        self.np_logits = np.ndarray((N, B, self.action_dim), dtype=np.float16, buffer=buf, offset=self.offset_logits)
        self.np_values = np.ndarray((N, B, 1), dtype=np.float32, buffer=buf, offset=self.offset_value)
        self.np_flags = np.ndarray((N,), dtype=np.uint8, buffer=buf, offset=self.offset_flags)

# =========================================================================
# Client Side: 伪装成 Model，供 MCTS 调用
# =========================================================================
class RemoteModel:
    def __init__(self, batcher: InferenceBatcher, worker_idx: int):
        self.batcher = batcher
        self.idx = worker_idx
        
    def __call__(self, board_tensor, ctx_tensor):
        """
        board_tensor: [B, 1, 20, 10]
        ctx_tensor:   [B, 11]
        """
        # 1. 检查 Batch Size 是否匹配
        batch_size = board_tensor.shape[0]
        # 如果 MCTS 在收尾阶段产生的小 batch，需要 pad 或者只填部分
        # 为简化逻辑，我们假设 MCTS 总是填满，或者是最大容量
        # 实际代码中，MCTS 的 batch_size 可能小于 max capacity，这里做个切片处理
        
        # Tensor -> Numpy (CPU)
        b_np = board_tensor.squeeze(1).numpy() # [B, 20, 10]
        c_np = ctx_tensor.numpy()              # [B, 11]
        
        # 2. 批量写入共享内存
        # 注意：这里我们写入 worker 对应的整个槽位
        # 如果 batch_size < self.batcher.local_batch_size，只写前部分
        self.batcher.np_boards[self.idx, :batch_size] = b_np
        self.batcher.np_ctxs[self.idx, :batch_size] = c_np
        
        # 3. 设置标志位
        self.batcher.np_flags[self.idx] = config.FLAG_REQ_READY
        
        # 4. 自旋等待
        while self.batcher.np_flags[self.idx] != config.FLAG_RES_READY:
             time.sleep(0.0001) # 极短 sleep 降低 CPU 占用
             pass
        
        # 5. 读取结果
        logits = self.batcher.np_logits[self.idx, :batch_size].astype(np.float32)
        value = self.batcher.np_values[self.idx, :batch_size]
        
        # 6. 重置标志
        self.batcher.np_flags[self.idx] = config.FLAG_IDLE
        
        # 7. 转回 Tensor
        t_logits = from_numpy(logits) # [B, 2304]
        t_values = from_numpy(value)  # [B, 1]
        
        return t_logits, t_values

    def eval(self): pass # 兼容接口
    def to(self, device): return self # 兼容接口

# =========================================================================
# Server Side: 跑在主进程或独立线程
# =========================================================================
def run_inference_loop(batcher: InferenceBatcher, model, device, stop_event):
    print("[Inference Server] Started with Parallel Batch Support.")
    
    # 预分配 pinned memory，注意维度变化
    # 总容量 = N * B
    total_capacity = batcher.num_workers * batcher.local_batch_size
    
    # 我们用 Flat 的缓冲区来接收数据，方便一次性送入模型
    pin_boards = zeros((total_capacity, 1, 20, 10), dtype=uint8).pin_memory()
    pin_ctxs = zeros((total_capacity, 11), dtype=float32).pin_memory()
    
    while not stop_event.is_set():
        # 1. 扫描 Ready 的 Worker
        ready_mask = (batcher.np_flags == config.FLAG_REQ_READY)
        indices = np.where(ready_mask)[0]
        
        num_ready = len(indices)
        if num_ready == 0:
            time.sleep(0.0001)
            continue
        
        # *优化*: 动态批处理等待
        # 如果 Ready 的只有 1 个且总数很多，稍微等一下让 Batch 变大（可选）
        if num_ready < batcher.num_workers * 0.5:
             # 极短的忙等，尝试凑单，可能会增加延迟但吞吐量大增
             for _ in range(1000): 
                 if np.sum(batcher.np_flags == config.FLAG_REQ_READY) > num_ready: break
             # 重新获取 indices
             ready_mask = (batcher.np_flags == config.FLAG_REQ_READY)
             indices = np.where(ready_mask)[0]
        
        # 2. 收集数据 (Batching)
        # 这里的难点是把 [K, B, ...] 变成 [K*B, ...]
        # 直接利用 numpy 的高级索引
        
        # raw_boards: [K, B, 20, 10]
        raw_boards = batcher.np_boards[indices] 
        raw_ctxs = batcher.np_ctxs[indices]
        
        # Flatten: [K*B, 20, 10]
        flat_boards = raw_boards.reshape(-1, 20, 10)
        flat_ctxs = raw_ctxs.reshape(-1, 11)
        
        # 3. 数据上 GPU
        t_boards = as_tensor(flat_boards).unsqueeze(1).to(device, non_blocking=True).float()
        t_ctxs = as_tensor(flat_ctxs).to(device, non_blocking=True)
        
        # 4. 推理
        with inference_mode():
             # 如果你的环境必须不用 autocast，这里去掉
             logits, values = model(t_boards, t_ctxs)
        
        # 5. 写回
        # Logits: [K*B, ActDim] -> Reshape -> [K, B, ActDim]
        out_logits = logits.cpu().numpy().astype(np.float16)
        out_values = values.cpu().numpy()
        
        out_logits = out_logits.reshape(len(indices), batcher.local_batch_size, -1)
        out_values = out_values.reshape(len(indices), batcher.local_batch_size, -1)
        
        batcher.np_logits[indices] = out_logits
        batcher.np_values[indices] = out_values
        
        # 6. 通知
        batcher.np_flags[indices] = config.FLAG_RES_READY