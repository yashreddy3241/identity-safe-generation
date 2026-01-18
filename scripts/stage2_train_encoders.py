import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.encoders import MultiFactorEncoder
import os

def hsic_loss(z1, z2):
    """
    Hilbert-Schmidt Independence Criterion to penalize dependence between factors.
    Simple kernel-based implementation.
    """
    # z1, z2: (B, D)
    def kernel_matrix(x, sigma=1.0):
        x_norm = x.pow(2).sum(1).view(-1, 1)
        k = torch.exp(-(x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())) / (2 * sigma**2))
        return k

    n = z1.size(0)
    k1 = kernel_matrix(z1)
    k2 = kernel_matrix(z2)
    
    # Centering matrix
    H = torch.eye(n, device=z1.device) - torch.ones((n, n), device=z1.device) / n
    
    # HSIC = tr(K1 H K2 H) / (n-1)^2
    loss = torch.trace(torch.mm(torch.mm(k1, H), torch.mm(k2, H))) / ((n - 1) ** 2)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5) # Short for prototype
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = torch.load(args.factors_path)
    # We need the images too, but for this simplified "Student" training, 
    # we usually need Image -> Encoder -> PredFactor vs RealFactor.
    # Ah, I need a dataloader that yields (Image, DictionaryOfFactors).
    # Since I saved factors separately, I need to align them.
    # For now, let's assume we can load images again.
    # To keep it simple, I will RELY on the fact that I saved 'filenames' in the .pt
    # and use dataset logic to reload images.
    
    # Re-instantiate dataset ?
    # Better: create a custom dataset wrapping the files and tensors.
    
    # Extract tensors
    # geometry: (N, 136), symmetry: (N, 193), texture: (N, 64), local: (N, 6)
    geom = data['geometry'].to(device)
    sym = data['symmetry'].to(device)
    tex = data['texture'].to(device)
    loc = data['local'].to(device)
    filenames = data['filenames']
    
    # Assume images are in data/raw/dummy_faces (Hardcoded for prototype simplicity or inferred)
    # We should probably pass image dir.
    # Let's just create a dummy "Image" tensor here if we can't find them, but strictly we need real images.
    # I'll rely on a 'image_dir' arg.
    image_dir = "data/raw/dummy_faces" # Mock assumption
    
    # Check dimensions
    dims = {
        'geometry': geom.shape[1],
        'symmetry': sym.shape[1],
        'texture': tex.shape[1],
        'local': loc.shape[1]
    }
    
    model = MultiFactorEncoder(dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mse = nn.MSELoss()
    
    # Dummy training loop on "features" ?? No we need images as input.
    # We need to load images.
    from src.dataset import UnlabeledFaceDataset
    dataset = UnlabeledFaceDataset(image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Ensure dataset length matches factors
    if len(dataset) != geom.shape[0]:
        print(f"Mismatch: Dataset {len(dataset)} vs Factors {geom.shape[0]}")
        return

    print("Training encoders...")
    model.train()
    for epoch in range(args.epochs):
        curr_idx = 0
        total_loss = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)
            
            # Slice factors
            batch_geom = geom[curr_idx : curr_idx+B]
            batch_sym = sym[curr_idx : curr_idx+B]
            batch_tex = tex[curr_idx : curr_idx+B]
            batch_loc = loc[curr_idx : curr_idx+B]
            
            curr_idx += B
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Reconstruction losses
            l_geom = mse(preds['geometry'], batch_geom)
            l_sym = mse(preds['symmetry'], batch_sym)
            l_tex = mse(preds['texture'], batch_tex)
            l_loc = mse(preds['local'], batch_loc)
            
            recon_loss = l_geom + l_sym + l_tex + l_loc
            
            # Disentanglement (HSIC)
            # Enforce independence between factors
            # e.g., geometry vs texture
            hsic_gt = hsic_loss(preds['geometry'], preds['texture'])
            # Add others as needed
            
            loss = recon_loss + 0.1 * hsic_gt
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss:.4f}")
        
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    main()
