import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src.model import FreqDetectNet
import yaml

def generate_visuals(image_path, model_path, config_path, output_path='results/visuals'):
    os.makedirs(output_path, exist_ok=True)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FreqDetectNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 1. Load Fake Image (Input)
    fake_img = cv2.imread(image_path)
    if fake_img is None: return
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
    fake_img = cv2.resize(fake_img, (cfg['data']['image_size'], cfg['data']['image_size']))
    
    # 2. Try to find Real pair (for the "Ground Truth" comparison)
    # Assume same filename in 'real' folder if current is 'fake'
    filename = os.path.basename(image_path)
    parent = os.path.dirname(os.path.dirname(image_path)) # data/test
    real_path = os.path.join(parent, 'real', filename)
    real_img = None
    
    if os.path.exists(real_path):
        real_img = cv2.imread(real_path)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        real_img = cv2.resize(real_img, (cfg['data']['image_size'], cfg['data']['image_size']))
    
    # 3. Model Inference
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = fake_img / 255.0
    img_tensor = (img_tensor - mean) / std
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, mask_logits = model(img_tensor)
        
    mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    binary_mask = (mask_prob > 0.5).astype(np.float32)
    
    # 4. Create Heatmap Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * binary_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(fake_img, 0.6, heatmap, 0.4, 0)
    
    # 5. Plot (5 Columns if pair exists, else 3)
    cols = 5 if real_img is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
    
    if real_img is not None:
        # Calculate Difference Mask (Ground Truth-ish)
        diff = np.mean(np.abs(fake_img.astype(float) - real_img.astype(float)), axis=2)
        gt_mask = (diff > 10).astype(float)
        
        axes[0].imshow(real_img)
        axes[0].set_title("Real Source")
        
        axes[1].imshow(fake_img)
        axes[1].set_title("Deepfake Input")
        
        axes[2].imshow(gt_mask, cmap='gray')
        axes[2].set_title("Ground Truth (Diff)")
        
        axes[3].imshow(binary_mask, cmap='gray')
        axes[3].set_title("Pred Binary Mask")
        
        axes[4].imshow(overlay)
        axes[4].set_title("Heatmap Overlay")
    else:
        axes[0].imshow(fake_img)
        axes[0].set_title("Deepfake Input")
        
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title("Pred Binary Mask")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Heatmap Overlay")
        
    for ax in axes: ax.axis('off')
    
    save_path = os.path.join(output_path, f"vis_{filename}")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to image file")
    parser.add_argument('--model', type=str, default='results/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    generate_visuals(args.image, args.model, args.config)
