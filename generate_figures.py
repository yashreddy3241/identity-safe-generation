import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src.model import FreqDetectNet
from src.transforms import DCTTransform
import yaml

def generate_visuals(image_path, model_path, config_path, output_path='results/visuals'):
    os.makedirs(output_path, exist_ok=True)
    
    # Load Config & Model
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FreqDetectNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process Image
    # 1. Load RGB
    raw_img = cv2.imread(image_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (cfg['data']['image_size'], cfg['data']['image_size']))
    
    # Normalize for Model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = raw_img / 255.0
    img_tensor = (img_tensor - mean) / std
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # 2. Forward Pass
    with torch.no_grad():
        cls_logits, mask_logits = model(img_tensor)
        dataset_dct = DCTTransform()(raw_img) # Just for visualization comparison
        
    # 3. Process Outputs
    pred_prob = torch.sigmoid(cls_logits).item()
    pred_mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
    
    # 4. Generate Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # A. Original Image
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input Image\nPred: {'FAKE' if pred_prob > 0.5 else 'REAL'} ({pred_prob:.2f})")
    axes[0].axis('off')
    
    # B. Frequency Spectrum (DCT)
    axes[1].imshow(dataset_dct, cmap='inferno')
    axes[1].set_title("Frequency Spectrum (DCT)")
    axes[1].axis('off')
    
    # C. Predicted Mask (Localization)
    axes[2].imshow(pred_mask, cmap='jet')
    axes[2].set_title("Predicted Trace Mask")
    axes[2].axis('off')
    
    # D. Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * pred_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Localization)")
    axes[3].axis('off')
    
    save_path = os.path.join(output_path, f"vis_{os.path.basename(image_path)}")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to image file")
    parser.add_argument('--model', type=str, default='results/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    generate_visuals(args.image, args.model, args.config)
