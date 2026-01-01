# ai/debug_overfit.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from ai.model import TetrisPolicyValue
from ai import config

def overfit_test():
    # 1. Load a tiny chunk of data
    print("Loading data...")
    with open(config.MEMORY_FILE, 'rb') as f:
        data = pickle.load(f)
    
    # 取前 64 个样本
    batch_size = 64
    b_boards = torch.tensor(data['boards'][:batch_size], dtype=torch.float32).to(config.DEVICE)
    b_ctxs   = torch.tensor(data['ctxs'][:batch_size], dtype=torch.float32).to(config.DEVICE)
    b_probs  = torch.tensor(data['probs'][:batch_size], dtype=torch.float32).to(config.DEVICE)
    b_values = torch.tensor(data['values'][:batch_size], dtype=torch.float32).to(config.DEVICE)

    # 增加维度适配模型 (B, 1, 20, 10)
    if b_boards.dim() == 3:
        b_boards = b_boards.unsqueeze(1)
    
    print(f"Input Shape: {b_boards.shape}")
    
    # 2. Init Model
    model = TetrisPolicyValue().to(config.DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Overfit Loop (Goal: Loss -> 0)...")
    
    for i in range(501):
        optimizer.zero_grad()
        
        logits, values = model(b_boards, b_ctxs)
        
        # Policy Loss (Cross Entropy)
        # b_probs 是 one-hot 的目标分布
        log_probs = F.log_softmax(logits, dim=1)
        p_loss = -torch.sum(b_probs * log_probs, dim=1).mean()
        
        # Value Loss (MSE)
        v_loss = F.mse_loss(values, b_values)
        
        loss = p_loss + v_loss
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            # Check accuracy
            pred_actions = torch.argmax(logits, dim=1)
            true_actions = torch.argmax(b_probs, dim=1)
            acc = (pred_actions == true_actions).float().mean()
            
            print(f"Iter {i:03d} | Loss: {loss.item():.6f} (P: {p_loss:.4f}, V: {v_loss:.4f}) | Acc: {acc:.2%}")

if __name__ == "__main__":
    overfit_test()