# ai/model.py (rewritten as Policy-Value network)
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
class TetrisPolicyValue(nn.Module):
    def __init__(self, num_res_blocks=4, action_dim=80, context_dim=10):
        super().__init__()
       
        # 输入通道: 只用 Board plane (1 channel)
        self.input_channels = 1
        self.action_dim = action_dim
       
        self.conv_input = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
       
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_res_blocks)
        ])
       
        # Context processing (Current, Hold, Next queue, Meta)
        # Context size: 10 (current, hold, next1-5, b2b, ren, can_hold)
        self.fc_context = nn.Linear(context_dim, 64)
       
        # Policy Head (logits over action_dim=80)
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1)  # Reduce channels
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc1 = nn.Linear(4 * 20 * 10 + 64, 512)
        self.policy_fc2 = nn.Linear(512, self.action_dim)
       
        # Value Head (scalar [-1, 1])
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 20 * 10 + 64, 128)
        self.value_fc2 = nn.Linear(128, 1)
    def forward(self, board, context):
        """
        board: [Batch, 20, 10] (Height x Width)
        context: [Batch, 10]
        Returns: logits [Batch, 80], value [Batch, 1]
        """
        batch_size = board.shape[0]
       
        # 构造输入图像 [B, 1, 20, 10]
        x = board.unsqueeze(1).float()  # [B, 1, 20, 10]
       
        # Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
           
        # Context embedding
        ctx_emb = F.relu(self.fc_context(context.float()))
       
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(batch_size, -1)
        p = torch.cat([p, ctx_emb], dim=1)
        p = F.relu(self.policy_fc1(p))
        logits = self.policy_fc2(p)  # Raw logits, no softmax
       
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(batch_size, -1)
        v = torch.cat([v, ctx_emb], dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # [-1, 1]
       
        return logits, value
