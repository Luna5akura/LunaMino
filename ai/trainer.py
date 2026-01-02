# ai/trainer.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import shutil
from datetime import datetime
from .model import TetrisPolicyValue
from .buffer import NumpyReplayBuffer
from . import config

class TetrisTrainer:
    def __init__(self, device=config.DEVICE):
        self.device = device
        
        # 模型初始化
        self.model = TetrisPolicyValue().to(device)
        # 内存格式优化 (NHWC)
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # 编译模型 (PyTorch 2.0+)
        # reduce-overhead 适合小 batch 快速迭代，max-autotune 适合吞吐量
        # if hasattr(torch, 'compile'):
        #     print("Compiling model with torch.compile...")
        #     self.model = torch.compile(self.model, mode='reduce-overhead')

        # 优化器优化: fused=True (仅在 CUDA 可用时)
        use_fused = 'cuda' in device and hasattr(optim.AdamW, 'has_fused')
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LR, 
            weight_decay=1e-4,
            fused=use_fused
        )
        
        self.scaler = GradScaler()
        self.buffer = NumpyReplayBuffer(config.MEMORY_SIZE)
        self.game_idx = 0

    def load_checkpoint(self, ui, force_reset=False):
        if force_reset:
            ui.log("[bold red]Force reset. Deleting old memory.[/bold red]")
            if os.path.exists(config.MEMORY_FILE): os.remove(config.MEMORY_FILE)
            if os.path.exists(config.CHECKPOINT_FILE): os.remove(config.CHECKPOINT_FILE)
            return 0
            
        if os.path.exists(config.CHECKPOINT_FILE):
            ui.log(f"Loading {config.CHECKPOINT_FILE}...")
            try:
                ckpt = torch.load(config.CHECKPOINT_FILE, map_location=self.device)
                # 注意：如果使用了 torch.compile，state_dict 的 key 可能会有 '_orig_mod' 前缀
                # 这里通常 PyTorch 会自动处理，但需留意
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.game_idx = ckpt.get('game_idx', 0)
            except Exception as e:
                ui.log(f"[red]Ckpt load failed: {e}[/red]")
        
        if os.path.exists(config.MEMORY_FILE):
             self.buffer.load(config.MEMORY_FILE, ui)
        return self.game_idx

    def save_checkpoint(self, ui, backup=False):
        ui.log(f"Saving game {self.game_idx}...")
        
        # 获取原始模型参数 (解包 compile 包装)
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_idx': self.game_idx
        }, config.CHECKPOINT_FILE)
        
        self.buffer.save(config.MEMORY_FILE, ui)
        if backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(config.CHECKPOINT_FILE, os.path.join(config.BACKUP_DIR, f"ckpt_{ts}.pth"))

    def update_weights(self, batch_size=config.BATCH_SIZE):
        if self.buffer.size < batch_size: return 0.0
        
        # 1. 采样
        b_b, b_c, b_p, b_v = self.buffer.sample(batch_size)
        
        # 2. 传输到 GPU + 强制转为 Float32
        # 注意：这里务必确保是 .float()，不能是 .half()
        t_b = torch.from_numpy(b_b).to(self.device, non_blocking=True).float()
        t_b = t_b.contiguous(memory_format=torch.channels_last)
        
        t_c = torch.from_numpy(b_c).to(self.device, non_blocking=True).float() # 确保 Context 也是 float
        t_p = torch.from_numpy(b_p).to(self.device, non_blocking=True).float()
        t_v = torch.from_numpy(b_v).to(self.device, non_blocking=True).float() # 确保 Value 也是 float
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # ==========================================================
        # 修改：暂时禁用 autocast，解决 NixOS 环境下的 cuDNN 兼容问题
        # ==========================================================
        
        # 原代码:
        # with autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
        #     logits, val = self.model(t_b, t_c)
        #     loss_policy = F.cross_entropy(logits, t_p)
        #     loss_value = F.mse_loss(val, t_v)
        #     loss = loss_policy + loss_value
        
        # 修改后 (纯 Float32 模式):
        logits, val = self.model(t_b, t_c)
        loss_policy = F.cross_entropy(logits, t_p)
        loss_value = F.mse_loss(val, t_v)
        loss = loss_policy + loss_value
            
        # ==========================================================
        
        # 3. 反向传播
        # 如果禁用了 autocast，GradScaler 其实也可以不用，但保留着也没坏处
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()