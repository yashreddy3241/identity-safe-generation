import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, jaccard_score
import argparse
import yaml
from tqdm import tqdm
from src.model import FreqDetectNet
from src.dataset import DeepfakeDataset

def evaluate(model_path, config_path):
    # Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    test_ds = DeepfakeDataset(cfg['data']['test_dir'], mode='val')
    test_loader = DataLoader(test_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    
    # Load Model
    model = FreqDetectNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Metrics
    all_preds = []
    all_dates = []
    all_masks = []
    all_mask_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            rgb = batch['rgb'].to(device)
            targets = batch['label'].cpu().numpy()
            
            cls_logits, mask_logits = model(rgb)
            
            # Classification
            probs = torch.sigmoid(cls_logits).cpu().numpy()
            all_preds.extend(probs)
            all_dates.extend(targets)
            
            # Segmentation (IoU)
            # Threshold at 0.5
            mask_pred = (torch.sigmoid(mask_logits) > 0.5).cpu().numpy()
            # Flatten for simple IoU calc over dataset or per image
            # Here doing global flat IoU for simplicity
            all_mask_preds.extend(mask_pred.flatten())
            all_masks.extend(batch['mask'].cpu().numpy().flatten())
            
    # Compute Final Metrics
    acc = accuracy_score(all_dates, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_dates, all_preds)
    f1 = f1_score(all_dates, [1 if p > 0.5 else 0 for p in all_preds])
    iou = jaccard_score(all_masks, all_mask_preds, average='macro') # Macro to balance 0/1 pixels
    
    print(f"\n=== SOTA Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean IoU: {iou:.4f} (Localization Score)")
    print(f"===============================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    evaluate(args.model, args.config)
