import os
import numpy as np
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num} dummy images in {args.output_dir}...")
    for i in range(args.num):
        # Random noise image 256x256
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(args.output_dir, f"dummy_{i:04d}.png"))
        
    print("Done.")

if __name__ == "__main__":
    main()
