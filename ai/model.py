# ai/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        # 1. CNN for Board
        self.conv1 = nn.Conv2d(1, CNN_CHANNELS[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=3, padding=1)
        
        # 自动计算 CNN 输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
            x = F.relu(self.conv1(dummy_input))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            cnn_out_size = x.view(1, -1).size(1)
            
        print(f"Model Debug: CNN Output Size {cnn_out_size}")

        # 2. Context
        self.ctx_fc = nn.Linear(CTX_DIM, EMBED_DIM)
        
        # 3. State Embedding (CNN + Context -> Hidden)
        # 注意：这里不再需要 Action Embedding 了！
        self.state_fc = nn.Linear(cnn_out_size + EMBED_DIM, HIDDEN_DIM)
        
        # 4. Output (Value)
        self.output_fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output_fc2 = nn.Linear(HIDDEN_DIM, 1)  # Scalar Value

    def forward(self, board, ctx):
        # Board: (Batch, 20, 10) -> (Batch, 1, 20, 10)
        if board.dim() == 3:
             board = board.unsqueeze(1)
             
        x = F.relu(self.conv1(board))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) # Flatten
        
        # Context
        c = F.relu(self.ctx_fc(ctx))
        
        # Combined: Board features + Game Context
        combined = torch.cat([x, c], dim=1)
        
        # Value Head
        v = F.relu(self.state_fc(combined))
        v = F.relu(self.output_fc1(v))
        q = self.output_fc2(v)
        
        return q