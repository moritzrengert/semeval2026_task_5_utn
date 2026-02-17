'''
Created by Noas Shaalan,

This script is used to train the MLP model for the ARES expert.

BEFORE USING THIS SCRIPT
1. Ares Context Features: must have been extracted using the prep_features_final.py script
2. You shouhld have the pre-trained vectors installed from sapeinza NLP lab: https://nlp.uniroma1.it/sensembert/


'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

from coral_head import CoralHead
from losses import coral_loss

FEATURES_FILE = os.path.join(SCRIPT_DIR, 'ares_context_features.pt')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'ares_coral_model.pth')

HYPERPARAMS = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'hidden_dim': 512,
    'weight_decay': 1e-4,
    'num_classes': 5,
    'dropout': 0.1
}

def run_epoch(model, loader, optimizer, device, is_training=True):
    model.train() if is_training else model.eval()
    running_loss = 0.0
    
    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    
    with context_manager:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_training: optimizer.zero_grad()
            
            logits = model(x)
            loss = coral_loss(logits, y - 1.0, num_classes=HYPERPARAMS['num_classes'], reduction='sum')
            
            if is_training:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
    return running_loss / len(loader.dataset)

def main():
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found. Run prep_features_final.py first.")
        return

    data = torch.load(FEATURES_FILE)
    train_x, train_y = data['train_features'], data['train_labels']
    dev_x, dev_y = data['dev_features'], data['dev_labels']
    
    train_y = torch.clamp(torch.round(train_y), 1.0, 5.0)

    mean, std = train_x.mean(dim=0), train_x.std(dim=0)
    std[std == 0] = 1.0
    train_x = (train_x - mean) / std
    dev_x = (dev_x - mean) / std 
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=HYPERPARAMS['batch_size'], shuffle=True)
    dev_loader = DataLoader(TensorDataset(dev_x, dev_y), batch_size=HYPERPARAMS['batch_size'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoralHead(
        input_dim=train_x.shape[1], 
        num_classes=HYPERPARAMS['num_classes'],
        hidden_dim=HYPERPARAMS['hidden_dim'],
        dropout=HYPERPARAMS['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'])
    
    best_val_loss = float('inf')
    
    print(f"Training ARES expert on {device}...")
    for epoch in range(HYPERPARAMS['epochs']):
        train_loss = run_epoch(model, train_loader, optimizer, device, is_training=True)
        dev_loss = run_epoch(model, dev_loader, None, device, is_training=False)
        
        if dev_loss < best_val_loss:
            best_val_loss = dev_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            status = "*"
        else:
            status = ""
            
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} {status}")
            
    print(f"\nTraining finished. Best Dev Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
