import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import numpy as np

# Fix Import Path for 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DeepfakeDataset
from torch.utils.data import DataLoader
from src.model import FreqDetectNet
import yaml

def evaluate_full_metrics(model_path, config_path, output_dir='results/metrics'):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 1. Plot Training History (from CSV)
    log_path = os.path.join(cfg['logging']['log_dir'], 'training_log.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.plot(df['epoch'], df['val_f1'], label='Val F1')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'training_graphs.png'))
        print("Generated Training Graphs.")
    else:
        print("No training log found. Skipping graphs.")

    # 2. Confusion Matrix (Real vs Fake)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FreqDetectNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    test_ds = DeepfakeDataset(cfg['data']['test_dir'], mode='val')
    # If using dummy generator, ensure test data exists or fallback
    if len(test_ds) == 0: return

    test_loader = DataLoader(test_ds, batch_size=cfg['data']['batch_size'], shuffle=False)
    
    y_true = []
    y_pred = []
    
    print("Generating Confusion Matrix...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            rgb = batch['rgb'].to(device)
            labels = batch['label'].cpu().numpy()
            
            cls_logits, _ = model(rgb)
            preds = (torch.sigmoid(cls_logits) > 0.5).cpu().numpy().flatten()
            
            y_true.extend(labels)
            y_pred.extend(preds)
            
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print("Generated Confusion Matrix.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    evaluate_full_metrics(args.model, args.config)
