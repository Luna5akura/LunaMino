# ai/model.py

import torch
import torch.nn as nn
from . import config

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 优化 1: 使用 1x1 卷积代替 Linear，避免 view/reshape 操作
        # 保持数据格式为 (B, C, 1, 1)
        mid_channels = channels // reduction
        self.se = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 优化 2: 使用 mean 代替 AdaptiveAvgPool2d，速度更快
        # x: (B, C, H, W) -> mean -> (B, C, 1, 1)
        y = x.mean((2, 3), keepdim=True)
        y = self.se(y)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
        
        # 优化 3: 显式 In-place 加法，利于算子融合
        out.add_(residual)
        out = self.relu(out)
        return out

class TetrisPolicyValue(nn.Module):
    def __init__(self, num_res_blocks=6, channels=32, action_dim=config.ACTION_DIM, context_dim=11):
        super().__init__()
        
        # 骨干网络
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 使用 ModuleList 并没有 Sequential 快，但在 export 时有时更灵活，这里保持 Sequential
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(channels, use_se=True) for _ in range(num_res_blocks)
        ])
        
        self.fc_ctx = nn.Sequential(
            nn.Linear(context_dim, 32), 
            nn.ReLU(inplace=True)
        )
        
        # -----------------------------------------------------
        # Policy Head
        # -----------------------------------------------------
        self.p_head_channels = 16
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, self.p_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.p_head_channels),
            nn.ReLU(inplace=True)
        )
        
        # 预计算 flatten 后的大小: 16 * 20 * 10 = 3200
        p_fc_input_dim = self.p_head_channels * 200 + 32
        
        self.p_fc = nn.Sequential(
            nn.Linear(p_fc_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, action_dim)
        )
        
        # -----------------------------------------------------
        # Value Head
        # -----------------------------------------------------
        self.v_head_channels = 2
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, self.v_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.v_head_channels),
            nn.ReLU(inplace=True)
        )
        
        # 预计算: 2 * 20 * 10 = 400
        v_fc_input_dim = self.v_head_channels * 200 + 32
        
        self.v_fc = nn.Sequential(
            nn.Linear(v_fc_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, board, context):
        # 建议: 外部调用时使用 with torch.inference_mode():
        
        x = self.conv_in(board)
        x = self.res_blocks(x)
        
        # 此时 x: (B, 64, 20, 10)
        
        # 处理 Context
        ctx_feat = self.fc_ctx(context) # (B, 32)
        
        # Policy Path
        p_x = self.p_conv(x)          # (B, 16, 20, 10)
        p_x = p_x.flatten(1)          # (B, 3200) - view 操作，零拷贝
        
        # 优化: 这里的 cat 不可避免，但由于维度较小，开销可控
        p_in = torch.cat([p_x, ctx_feat], dim=1)
        logits = self.p_fc(p_in)
        
        # Value Path
        v_x = self.v_conv(x)          # (B, 2, 20, 10)
        v_x = v_x.flatten(1)          # (B, 400)
        
        v_in = torch.cat([v_x, ctx_feat], dim=1)
        value = self.v_fc(v_in)
        
        return logits, value

    def to_efficient_memory_format(self):
        """
        辅助函数: 将模型转换为 Channels Last 格式 (NHWC)。
        在 NVIDIA GPU (Tensor Cores) 上通常能获得 20%+ 的加速。
        用法: model = TetrisPolicyValue().cuda().to_efficient_memory_format()
        """
        self.to(memory_format=torch.channels_last)
        return self