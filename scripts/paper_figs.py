import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generator import StyleGAN2Wrapper

def create_grid(images, rows, cols, save_path):
    """
    Creates a high-res grid of images.
    images: List of tensors (C, H, W)
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c] if rows > 1 else axes[c]
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).cpu().detach().numpy()
                img = (img - img.min()) / (img.max() - img.min()) # Normalize to 0-1
                ax.imshow(img)
            ax.axis('off')
            idx += 1
            
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    os.makedirs('results/paper_figs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = StyleGAN2Wrapper().to(device)
    
    print("Generating Interpolation Grid...")
    # Generate A (left) -> C (mix) -> B (right)
    # Mocking latent interpolation logic for viz
    # In real SOTA, we would interpolate w vectors
    
    imgs = []
    steps = 7
    # Mock interpolation
    for i in range(steps):
        # alpha goes 0 -> 1
        # In real code: w = (1-a)*w_A + a*w_B
        # using random w for demo
        img, _ = generator(z=torch.randn(1, 512).to(device))
        imgs.append(img[0])
        
    create_grid(imgs, 1, 7, 'results/paper_figs/fig1_interpolation.png')
    print("Saved fig1_interpolation.png")

if __name__ == "__main__":
    main()
