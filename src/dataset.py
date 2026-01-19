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
    Attempts to find paired Real image for Fakes to compute GT Mask.
    """
    def __init__(self, root_dir, image_size=256, transform=None, mode='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        self.dct_transform = DCTTransform()
        
        # Collect file paths
        self.real_paths = glob.glob(os.path.join(root_dir, 'real', '*.*'))
        self.fake_paths = glob.glob(os.path.join(root_dir, 'fake', '*.*'))
        self.all_paths = self.real_paths + self.fake_paths
        
        # Labels: 0 for Real, 1 for Fake
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        # Pair Lookup (Filename -> Real Path)
        self.real_lookup = {os.path.basename(p): p for p in self.real_paths}
        
        # Albumentations pipelines
        if transform:
            self.transform = transform
        else:
            if mode == 'train':
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        label = self.labels[idx]
        filename = os.path.basename(path)
        
        image = cv2.imread(path)
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        dct_map = self.dct_transform(image)
        dct_tensor = torch.from_numpy(dct_map).permute(2, 0, 1).float().repeat(3, 1, 1)
        
        # 2. GT Mask Generation
        # If Real: Mask is zeros.
        # If Fake: Try to find pair and diff. Else Ones.
        gt_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        if label == 1:
            if filename in self.real_lookup:
                real_path = self.real_lookup[filename]
                real_img = cv2.imread(real_path)
                if real_img is not None:
                    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                    real_img = cv2.resize(real_img, (self.image_size, self.image_size))
                    fake_img = cv2.resize(image, (self.image_size, self.image_size))
                    
                    # Compute Diff
                    diff = np.abs(fake_img.astype(np.float32) - real_img.astype(np.float32))
                    diff = np.mean(diff, axis=2) # Average channels
                    gt_mask = (diff > 10).astype(np.float32) # Threshold 10/255 intensity diff
                else:
                    gt_mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
            else:
                # Weak supervision fallback
                gt_mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
                
        # 3. Augmentations
        # Only normalize for now to keep it simple, or apply consistent augment to mask?
        # A.Compose handles mask augmentation if passed
        transformed = self.transform(image=image, mask=gt_mask)
        rgb_tensor = transformed['image']
        mask_tensor = transformed['mask'].unsqueeze(0).float() # (1, H, W)
        
        return {
            'rgb': rgb_tensor,
            'dct': dct_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'mask': mask_tensor,
            'path': path
        }
