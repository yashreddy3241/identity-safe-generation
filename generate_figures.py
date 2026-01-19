import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src.model import FreqDetectNet
import yaml
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Confusion Matrix to {save_path}")

def generate_visuals(image_path, model_path, config_path, output_path='results/visuals'):
    os.makedirs(output_path, exist_ok=True)
    
    # Load Config & Model
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FreqDetectNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Process Image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (cfg['data']['image_size'], cfg['data']['image_size']))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = raw_img / 255.0
    img_tensor = (img_tensor - mean) / std
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Forward Pass
    with torch.no_grad():
        cls_logits, mask_logits = model(img_tensor)
        
    # Process Outputs
    pred_prob = torch.sigmoid(cls_logits).item()
    pred_class = "Fake" if pred_prob > 0.5 else "Real"
    
    # Create Binary Mask (Threshold at 0.5)
    mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    binary_mask = (mask_prob > 0.5).astype(np.float32)
    
    # Figure Generation (Requested Format)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input: {pred_class} ({pred_prob:.1%})")
    axes[0].axis('off')
    
    # 2. Binary Mask (Black & White)
    axes[1].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Binary Mask (White=Fake)")
    axes[1].axis('off')
    
    # 3. Overlay (Red = Fake Regions)
    heatmap = np.zeros_like(raw_img)
    heatmap[:, :, 0] = (binary_mask * 255).astype(np.uint8) # Red Channel
    overlay = cv2.addWeighted(raw_img, 0.7, heatmap, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Localization Overlay")
    axes[2].axis('off')
    
    save_path = os.path.join(output_path, f"report_{os.path.basename(image_path)}")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Report to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to image file")
    parser.add_argument('--model', type=str, default='results/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    generate_visuals(args.image, args.model, args.config)
