# ai/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        # 1. 定义卷积层
        self.conv1 = nn.Conv2d(1, CNN_CHANNELS[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=3, padding=1)
        
        # 2. 自动计算 CNN 输出维度 (Dummy Pass)
        # 这一步非常关键，避免手动计算错误
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
            x = F.relu(self.conv1(dummy_input))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            # 计算扁平化后的大小
            cnn_out_size = x.view(1, -1).size(1)
            
        print(f"Model Debug: CNN Output Size calculated as {cnn_out_size} (Expected ~640)")

        # 3. 定义全连接层
        # Context Encoding
        self.ctx_fc = nn.Linear(CTX_DIM, EMBED_DIM)
        
        # State Embedding (CNN features + Context features)
        self.state_fc = nn.Linear(cnn_out_size + EMBED_DIM, HIDDEN_DIM)
        
        # Action Embedding
        self.action_fc1 = nn.Linear(ACTION_DIM, EMBED_DIM)
        self.action_fc2 = nn.Linear(EMBED_DIM, EMBED_DIM)
        
        # Combined Processing
        self.combined_fc1 = nn.Linear(HIDDEN_DIM + EMBED_DIM, HIDDEN_DIM)
        self.combined_fc2 = nn.Linear(HIDDEN_DIM, 1)  # Scalar Q-value

    def forward(self, board, ctx, action):
        # Board: bs x 1 x 20 x 10
        # 必须确保这里加上了 channel 维度 (unsqueeze(1))
        if board.dim() == 3: # (Batch, Height, Width) -> (Batch, 1, Height, Width)
             board = board.unsqueeze(1)
             
        x = F.relu(self.conv1(board))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) # Flatten
        
        # Context: bs x 11
        c = F.relu(self.ctx_fc(ctx))
        
        # State embed: Concatenate CNN out + Context Embed
        state_emb = F.relu(self.state_fc(torch.cat([x, c], dim=1)))
        
        # Action: bs x 4 (normalized)
        a = F.relu(self.action_fc1(action))
        a = F.relu(self.action_fc2(a))
        
        # Combined
        combined = torch.cat([state_emb, a], dim=1)
        combined = F.relu(self.combined_fc1(combined))
        q = self.combined_fc2(combined)
        
        return q