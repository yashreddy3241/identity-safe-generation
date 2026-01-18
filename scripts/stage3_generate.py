import argparse
import torch
import os
import yaml
from torchvision.utils import save_image
from src.generator import StyleGAN2Wrapper, RecompositionModule
from src.factors import FactorWrapper # To get dims

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='results/generated')
    parser.add_argument('--tau', type=float, default=0.5, help="Leakage threshold (checked in Stage 4)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Models
    # Mock dims from Stage 1
    # geometry: 136, symmetry: 193, texture: 64
    factor_dims = {'geometry': 136, 'symmetry': 193, 'texture': 64}
    
    recomposer = RecompositionModule(factor_dims).to(device)
    generator = StyleGAN2Wrapper().to(device)
    
    # Ideally load weights
    # recomposer.load_state_dict(...)
    
    # Generate Synthetic Factors (The "A vs B" mix)
    # For demo, we just sample random factors as "A" and "B" and mix them.
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i in range(args.num_samples):
            # Sample "Source A" and "Source B" factors (Synthetic)
            factors_A = {
                'geometry': torch.randn(1, 136).to(device),
                'symmetry': torch.randn(1, 193).to(device),
                'texture': torch.randn(1, 64).to(device)
            }
            factors_B = {
                'geometry': torch.randn(1, 136).to(device),
                'symmetry': torch.randn(1, 193).to(device),
                'texture': torch.randn(1, 64).to(device)
            }
            
            # Mix factors (Controlled Mixing)
            # w_f can be different per factor
            w_geom = 0.3 # More B-like geometry
            w_tex = 0.7  # More A-like texture
            
            factors_C = {
                'geometry': w_geom * factors_A['geometry'] + (1-w_geom) * factors_B['geometry'],
                'symmetry': 0.5 * factors_A['symmetry'] + 0.5 * factors_B['symmetry'],
                'texture': w_tex * factors_A['texture'] + (1-w_tex) * factors_B['texture']
            }
            
            # Recompose -> Latent
            w_latent = recomposer(factors_C)
            
            # Generate
            img, _ = generator(w=w_latent)
            
            # Save
            save_image(img, os.path.join(args.output_dir, f"sample_{i:04d}.png"), normalize=True, value_range=(-1, 1))
            
    print("Done.")

if __name__ == "__main__":
    main()
