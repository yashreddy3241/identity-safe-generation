import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import yaml
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from src.model import FreqDetectNet
from src.dataset import DeepfakeDataset

# Load Config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Loss Functions
class DualTaskLoss(nn.Module):
    def __init__(self, seg_weight=0.5):
        super(DualTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, cls_logits, mask_logits, cls_targets, mask_targets):
        cls_loss = self.cls_criterion(cls_logits, cls_targets.unsqueeze(1))
        seg_loss = self.seg_criterion(mask_logits, mask_targets)
        return cls_loss + self.seg_weight * seg_loss, cls_loss, seg_loss

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training")
    
    for batch in loop:
        rgb = batch['rgb'].to(device)
        targets = batch['label'].to(device)
        masks = batch['mask'].to(device)

        with autocast():
            cls_logits, mask_logits = model(rgb)
            loss, cls_l, seg_l = criterion(cls_logits, mask_logits, targets, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            rgb = batch['rgb'].to(device)
            targets = batch['label'].to(device)
            masks = batch['mask'].to(device)

            with autocast():
                cls_logits, mask_logits = model(rgb)
                loss, _, _ = criterion(cls_logits, mask_logits, targets, masks)

            total_loss += loss.item()
            probs = torch.sigmoid(cls_logits)
            preds.extend((probs > 0.5).cpu().numpy())
            truths.extend(targets.cpu().numpy())
            
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    return total_loss / len(loader), acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = DeepfakeDataset(cfg['data']['train_dir'], mode='train')
    val_ds = DeepfakeDataset(cfg['data']['test_dir'], mode='val')
    
    if len(train_ds) == 0:
        print("Warning: No data found.")
        return
        
    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True,
                              num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False,
                            num_workers=cfg['data']['num_workers'], pin_memory=True)

    model = FreqDetectNet(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
    scaler = GradScaler()
    criterion = DualTaskLoss()
    
    best_acc = 0.0
    history = []
    
    log_dir = cfg['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    for epoch in range(cfg['training']['epochs']):
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # Save Log
        pd.DataFrame(history).to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(cfg['training']['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['training']['save_dir'], 'best_model.pth'))
            print("Saved Best Model!")

if __name__ == '__main__':
    main()
