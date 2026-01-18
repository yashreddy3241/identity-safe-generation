import torch
import torch.nn as nn
import os
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generator import StyleGAN2Wrapper

# Optional Evaluation Imports
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not found.")

def compute_stability(generator, device, n_samples=100, perturbation=0.01):
    """
    Measure how much the image changes given small latent perturbations.
    Lower LPIPS distance = Higher Stability.
    """
    if not HAS_LPIPS:
        return 0.0
        
    print("Computing Stability Score...")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    total_dist = 0.0
    
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, 512).to(device)
            z_p = z + torch.randn_like(z) * perturbation
            
            img, _ = generator(z=z)
            img_p, _ = generator(z=z_p)
            
            # LPIPS expects [-1, 1]
            dist = loss_fn(img, img_p)
            total_dist += dist.item()
            
    return total_dist / n_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid_real_path', type=str, help="Path to real images for FID")
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = StyleGAN2Wrapper().to(device)
    
    # 1. Stability
    stab = compute_stability(generator, device, args.num_samples)
    print(f"Stability Score (LPIPS @ sigma=0.01): {stab:.4f}")
    
    # 2. FID (Stub)
    # To implement real FID, we would use torch-fidelity on 'fid_real_path' vs generated folder
    if args.fid_real_path and os.path.exists(args.fid_real_path):
        print("FID computation requires full 'torch-fidelity' run on generated directory.")
        print(f"Reference: {args.fid_real_path}")
        print("Please run: torch-fidelity --input1 results/generated --input2 {args.fid_real_path}")

if __name__ == "__main__":
    main()
