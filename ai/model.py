# ai/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TetrisPolicyValue(nn.Module):
    def __init__(self, num_res_blocks=5, action_dim=256, context_dim=11):  # More blocks
        super().__init__()
        self.action_dim = action_dim
        self.board_h, self.board_w = 20, 10
        
        filters = 128  # Increased for capacity
        
        # 1. Initial Conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
        # 2. Residual Tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(num_res_blocks)]
        )
        
        # 3. Policy Head
        self.p_conv = nn.Sequential(
            nn.Conv2d(filters, 4, kernel_size=1, bias=False),  # More channels
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        # Context fusion: deeper MLP
        self.p_fc_ctx = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.p_fc = nn.Sequential(
            nn.Linear(4 * self.board_h * self.board_w + 32, 512),  # Larger hidden
            nn.ReLU(),
            nn.Dropout(0.1),  # New: regularization
            nn.Linear(512, action_dim)
        )
        
        # 4. Value Head
        self.v_conv = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),  # More channels
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.v_fc_ctx = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.v_fc = nn.Sequential(
            nn.Linear(2 * self.board_h * self.board_w + 32, 256),  # Larger
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, board, context):
        x = board.unsqueeze(1)
        
        # Backbone
        x = self.conv_in(x)
        x = self.res_blocks(x)
        
        # Policy Head
        p_x = self.p_conv(x).flatten(1)
        p_ctx = self.p_fc_ctx(context)
        p_in = torch.cat([p_x, p_ctx], dim=1)
        logits = self.p_fc(p_in)
        
        # Value Head
        v_x = self.v_conv(x).flatten(1)
        v_ctx = self.v_fc_ctx(context)
        v_in = torch.cat([v_x, v_ctx], dim=1)
        value = self.v_fc(v_in)
        
        return logits, value