import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.transforms import DCTTransform

class DeepfakeDataset(Dataset):
    """
    Dual-Stream Dataset: Returns (RGB, DCT) and Label.
    Assumes structure:
       root/real/image.jpg
       root/fake/image.jpg
    """
    def __init__(self, root_dir, image_size=256, transform=None, mode='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        self.dct_transform = DCTTransform()
        
        # Collect file paths
        # We look for 'real' and 'fake' subfolders
        self.real_paths = glob.glob(os.path.join(root_dir, 'real', '*.*'))
        self.fake_paths = glob.glob(os.path.join(root_dir, 'fake', '*.*'))
        self.all_paths = self.real_paths + self.fake_paths
        
        # Labels: 0 for Real, 1 for Fake
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        # Albumentations pipelines
        if transform:
            self.transform = transform
        else:
            if mode == 'train':
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussNoise(p=0.2), # Robustness to noise
                    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3), # Robustness to jpeg
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                
        # DCT transform needs to happen before normalization (on raw pixels)
        # So we handle it separately inside __getitem__

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        label = self.labels[idx]
        
        # Read Image (BGR to RGB)
        image = cv2.imread(path)
        if image is None:
            # Handle broken images gracefully
            return self.__getitem__((idx + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Compute DCT *before* standard augmentations
        # We resize first to ensure standard DCT computation size
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        dct_map = self.dct_transform(image_resized) # Returns (H, W, 1) in 0-1
        
        # Convert DCT to tensor, repeat to 3 channels for backbone compatibility
        dct_tensor = torch.from_numpy(dct_map).permute(2, 0, 1).float() # (1, H, W)
        dct_tensor = dct_tensor.repeat(3, 1, 1) # (3, H, W)
        
        # 2. RGB Transforms (Augmentations)
        transformed = self.transform(image=image)
        rgb_tensor = transformed['image']
        
        # 3. Create Mask (For localization)
        # For now, we use a Weakly Supervised approach or Full Image Label if masks aren't standard.
        # But user mentioned localization.
        # If we don't have ground truth masks, we can map classification label to mask.
        # Fake = 1s, Real = 0s
        mask = torch.ones((1, self.image_size, self.image_size)) * label
        
        return {
            'rgb': rgb_tensor,
            'dct': dct_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'mask': mask.float(),
            'path': path
        }
