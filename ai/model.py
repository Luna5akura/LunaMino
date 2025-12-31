# ai/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 将 ResidualBlock 转换为 JIT Script 兼容写法
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class TetrisPolicyValue(nn.Module):
    def __init__(self, num_res_blocks=3, action_dim=256, context_dim=11):
        super().__init__()
        self.filters = 32
        
        # Backbone
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.filters),
            nn.ReLU(inplace=True)
        )
        
        # 使用 ModuleList 或 Sequential 均可，Sequential 对 JIT 更友好
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.filters) for _ in range(num_res_blocks)]
        )
        
        # Context MLP
        self.fc_ctx = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(inplace=True)
        )
        
        # --- Policy Head ---
        # 优化：Conv 减少通道 -> Flatten -> Concat -> MLP
        # 保持空间信息直到 Flatten
        self.p_conv = nn.Sequential(
            nn.Conv2d(self.filters, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        # Flatten size: 4 * 20 * 10 = 800
        self.p_fc = nn.Sequential(
            nn.Linear(800 + 32, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim)
        )
        
        # --- Value Head ---
        self.v_conv = nn.Sequential(
            nn.Conv2d(self.filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # Flatten size: 2 * 20 * 10 = 400
        self.v_fc = nn.Sequential(
            nn.Linear(400 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # 3. 显式权重初始化 (加速收敛)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight) # RL 中常用正交初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 2. 移除 dim 检查，配合 JIT 装饰器
    # 注意：在外部实例化模型后，建议调用 model = torch.jit.script(model)
    def forward(self, board, context):
        # 假设输入已经是 (B, 1, 20, 10)，移除了 unsqueeze 判断以提升速度
        # Backbone
        x = self.conv_in(board)
        x = self.res_blocks(x)
        
        # Context Feature (B, 32)
        ctx_feat = self.fc_ctx(context)
        
        # --- Policy Path ---
        p_x = self.p_conv(x)
        p_x = p_x.flatten(1) # (B, 800)
        # 显式拼接，避免不必要的中间变量
        logits = self.p_fc(torch.cat([p_x, ctx_feat], dim=1))
        
        # --- Value Path ---
        v_x = self.v_conv(x)
        v_x = v_x.flatten(1) # (B, 400)
        value = self.v_fc(torch.cat([v_x, ctx_feat], dim=1))
        
        return logits, value

# 辅助函数：用于 MCTS 初始化时确保模型处于最佳状态
def load_optimized_model(device):
    model = TetrisPolicyValue().to(device)
    # 开启 Channels Last 内存布局 (对 Tensor Core 友好)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    
    # 尝试 JIT 编译 (如果环境支持)
    # try:
    #     # 需要提供样例输入来 Trace，或者直接 Script
    #     # 对于包含动态逻辑较少的模型，script 更好
    #     model = torch.jit.script(model)
    # except Exception as e:
    #     print(f"JIT Compilation skipped: {e}")
        
    return model