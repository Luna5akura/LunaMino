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

class TetrisNet(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        
        # Input: 
        # Channel 0: Board (10x20)
        # Channel 1-7: One-hot encoded Current Piece Type (7 planes)
        self.conv_input = nn.Conv2d(1 + 7, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_res_blocks)
        ])
        
        # Context processing (Queue, Hold, Meta)
        # Context size from utils.py is approx 9 integers.
        self.fc_context = nn.Linear(9, 32) 
        
        # Policy Head
        # Output: Probability for every possible placement (x=0..9, rot=0..3) = 40 output logits
        # Note: Usually we output 40, and mask illegal ones. 
        # But wait, y coordinate? 
        # The AI only controls X and Rotation. Y is determined by gravity (Hard Drop).
        # So action space size is 10 (width) * 4 (rotations) = 40.
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 20 + 32, 80) # +32 for context
        
        # Value Head
        # Output: Scalar evaluation of the position (-1 to 1)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 10 * 20 + 32, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, context, piece_type):
        """
        board: [Batch, 10, 20] -> Needs unsqueeze channel
        context: [Batch, 9]
        piece_type: [Batch] (int 0-6)
        """
        batch_size = board.shape[0]
        
        # 1. Prepare Spatial Input
        x = board.unsqueeze(1).float() # [B, 1, 20, 10] (assuming numpy y,x order)
        
        # Create one-hot planes for piece type and broadcast to board size
        piece_plane = torch.zeros(batch_size, 7, 20, 10, device=board.device)
        for i in range(batch_size):
            p_idx = piece_type[i]
            if 0 <= p_idx < 7:
                piece_plane[i, p_idx, :, :] = 1
        
        x = torch.cat([x, piece_plane], dim=1) # [B, 8, 20, 10]
        
        # 2. Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # 3. Process Context
        ctx = F.relu(self.fc_context(context.float()))
        
        # 4. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(batch_size, -1) # Flatten
        p = torch.cat([p, ctx], dim=1)
        policy_logits = self.policy_fc(p) # [B, 40]
        
        # 5. Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(batch_size, -1)
        v = torch.cat([v, ctx], dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # [-1, 1]
        
        return policy_logits, value