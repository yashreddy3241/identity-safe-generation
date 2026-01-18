import os
import argparse
import torch
from tqdm import tqdm
from src.dataset import create_dataloader
from src.factors import FactorWrapper

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Factor Extraction")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input images")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save .pt dictionary of factors")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Data
    loader = create_dataloader(args.input_dir, batch_size=args.batch_size)
    if len(loader.dataset) == 0:
        print("No images found. Exiting.")
        return

    # Extractor
    extractor = FactorWrapper(device=device)

    # Storage
    all_factors = {
        'geometry': [],
        'symmetry': [],
        'texture': [],
        'local': []
    }
    filenames = []

    print("Extracting factors...")
    with torch.no_grad():
        for batch_imgs, batch_names in tqdm(loader):
            batch_imgs = batch_imgs.to(device)
            # factors is a dict of tensors
            factors = extractor.extract_all(batch_imgs)
            
            for k, v in factors.items():
                all_factors[k].append(v.cpu())
            
            filenames.extend(batch_names)

    # Concatenate
    final_dict = {
        'filenames': filenames
    }
    for k in all_factors:
        final_dict[k] = torch.cat(all_factors[k], dim=0)
        print(f"Factor {k}: shape {final_dict[k].shape}")

    # Save
    torch.save(final_dict, args.output_path)
    print(f"Saved factors to {args.output_path}")

if __name__ == "__main__":
    main()
