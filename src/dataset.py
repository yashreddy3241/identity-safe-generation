import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

class UnlabeledFaceDataset(Dataset):
    """
    Simple dataset for loading images from a directory.
    Assumes images are already roughly aligned or we interpret them as is.
    """
    def __init__(self, root_dir, transform=None, image_size=256):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')) + 
                                  glob.glob(os.path.join(root_dir, '*.jpg')) +
                                  glob.glob(os.path.join(root_dir, '*.jpeg')))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}. Dataset is empty.")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1] range commonly used for GANs
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, os.path.basename(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 256, 256), "error"

def create_dataloader(root_dir, batch_size=4, num_workers=0, image_size=256):
    dataset = UnlabeledFaceDataset(root_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
