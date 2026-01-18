import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add StyleGAN2 dependency to path for unpickling
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stylegan2-ada-pytorch')))

import pickle

class StyleGAN2Wrapper(nn.Module):
    """
    Wrapper for StyleGAN2-ADA.
    Supports both Mock (prototype) and Real (SOTA) weights.
    """
    def __init__(self, resolution=256, z_dim=512, w_dim=512, n_layers=14, weights_path=None):
        super().__init__()
        self.resolution = resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.n_layers = n_layers
        self.real_G = None
        
        # SOTA Upgrade: Load Real Weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading SOTA StyleGAN2 weights from {weights_path}...")
            try:
                with open(weights_path, 'rb') as f:
                    self.real_G = pickle.load(f)['G_ema'] # G_ema is standard for StyleGAN2-ADA
                self.real_G.eval()
                print("Success: Real Generator Loaded.")
            except Exception as e:
                print(f"Failed to load weights: {e}. Fallback to Mock.")
        
        # Fallback Mock Network
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim * n_layers) 
        )
        
        self.synth_main = nn.Sequential(
            nn.ConvTranspose2d(w_dim, 256, 4, 1, 0), # 4x4
            nn.ReLU(),
            nn.Upsample(scale_factor=4), # 16x16
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4), # 64x64
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4), # 256x256
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, z=None, w=None, truncation_psi=0.7, noise_mode='const'):
        # 1. Use Real Generator if available
        if self.real_G is not None:
            # Real StyleGAN2-ADA signature: (z, c, truncation_psi, noise_mode)
            # Or synthesis(w, noise_mode)
            # We need to map our inputs carefully.
            
            if w is not None:
                # W-space input (B, 14, 512)
                # Official G.synthesis accepts w
                return self.real_G.synthesis(w, noise_mode=noise_mode), w
            elif z is not None:
                # Z-space input (B, 512)
                # Official G(z, c)
                # c is label, usually valid zero tensor or None
                img = self.real_G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
                return img, None # w is hidden inside G call unless we separate mapping
                
        # 2. Mock Fallback
        if w is None:
            if z is None:
                raise ValueError("Must provide z or w")
            w = self.mapping(z) # (B, w_dim * n_layers)
            w = w.view(-1, self.n_layers, self.w_dim)
            
        w_avg = w.mean(dim=1)
        w_spatial = w_avg.view(-1, self.w_dim, 1, 1)
        img = self.synth_main(w_spatial)
        return img, w

class RecompositionModule(nn.Module):
    """
    Maps factor vectors {z_f} to StyleGAN latent space W+.
    """
    def __init__(self, factor_dims, w_dim=512, n_layers=14):
        super().__init__()
        self.n_layers = n_layers
        self.w_dim = w_dim
        
        # Define which layers each factor controls (Heuristic split)
        # Geometry: Coarse (0-3)
        # Symmetry: (Shared/Global)
        # Texture: Fine (4-13)
        # Local: (Ignored for global W map, requires spatial inputs, implicitly handled by W)
        
        self.factor_dims = factor_dims
        
        # Projectors
        self.proj_geom = nn.Linear(factor_dims['geometry'], w_dim * 4) # Layers 0-3
        self.proj_tex = nn.Linear(factor_dims['texture'], w_dim * (n_layers - 4)) # Layers 4-end
        
        # Symmetry might modulate geometry
        self.proj_sym = nn.Linear(factor_dims['symmetry'], w_dim * 4)
        
    def forward(self, factors):
        # factors: dict
        z_g = factors['geometry']
        z_t = factors['texture']
        z_s = factors['symmetry']
        
        B = z_g.size(0)
        
        # Map geometry -> W coarse
        w_coarse_g = self.proj_geom(z_g).view(B, 4, self.w_dim)
        w_coarse_s = self.proj_sym(z_s).view(B, 4, self.w_dim)
        
        # Combine (simple add or gate)
        w_coarse = w_coarse_g + 0.1 * w_coarse_s
        
        # Map texture -> W fine
        w_fine = self.proj_tex(z_t).view(B, self.n_layers - 4, self.w_dim)
        
        # Concat
        w = torch.cat([w_coarse, w_fine], dim=1) # (B, 14, 512)
        
        return w
