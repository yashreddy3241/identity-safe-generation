import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

# Try SOTA imports
try:
    import face_alignment
    HAS_FAN = True
except ImportError:
    HAS_FAN = False
    print("Warning: face_alignment not found. Using Mock Geometry.")

try:
    from facenet_pytorch import InceptionResnetV1
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False
    print("Warning: facenet_pytorch not found. Using Mock Texture.")

class BaseFactorExtractor(ABC):
    def __init__(self, device='cpu'):
        self.device = device
        
    @abstractmethod
    def extract(self, images):
        pass

class GeometryExtractor(BaseFactorExtractor):
    """
    Extracts geometric information.
    SOTA: Uses generic Face Alignment Network (FAN) for 68 landmarks.
    """
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.fan = None
        if HAS_FAN:
            try:
                # 2D landmarks for speed, '3d' for more detail
                self.fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=str(device))
                print("Success: FAN Geometry Loaded.")
            except Exception as e:
                print(f"Failed to init FAN: {e}. Fallback to mock.")

    def extract(self, images):
        # images: (B, 3, H, W)
        B = images.shape[0]
        
        if self.fan is not None:
            # FAN expects numpy (H, W, 3) usually or batch
            # This is slow per-image loop in prototype, batch inference preferred if supported
            preds = []
            for i in range(B):
                img_np = images[i].permute(1, 2, 0).cpu().detach().numpy()
                # Denormalize if needed? FAN expects 0-255 usually?
                # Assuming images are [-1, 1], map to [0, 255]
                img_np = ((img_np + 1) * 127.5).astype(np.uint8)
                
                try:
                    landmarks = self.fan.get_landmarks(img_np)
                    if landmarks:
                        lm = torch.tensor(landmarks[0]).flatten() # (68*2)
                        # Resize/Pad if standard is fixed
                    else:
                        lm = torch.zeros(136)
                except:
                    lm = torch.zeros(136)
                preds.append(lm)
            
            return torch.stack(preds).to(self.device)
            
        # Mock Fallback
        return torch.randn(B, 136, device=self.device)

class SymmetryExtractor(BaseFactorExtractor):
    """
    SOTA Upgrade: Pixel-wise difference on aligned faces + Feature distance.
    """
    def extract(self, images):
        images_flipped = torch.flip(images, dims=[3])
        diff = torch.abs(images - images_flipped)
        
        # 1. Pixel stats
        mean_diff = torch.mean(diff, dim=[2, 3]) 
        global_sym = torch.mean(mean_diff, dim=1, keepdim=True)
        
        # 2. Detailed grid
        diff_small = F.interpolate(diff, size=(8, 8), mode='bilinear', align_corners=False)
        flat_diff = diff_small.view(images.shape[0], -1) 
        
        return torch.cat([global_sym, flat_diff], dim=1)

class TextureExtractor(BaseFactorExtractor):
    """
    SOTA Upgrade: Uses InceptionResnetV1 deep features (mid-level) for texture/style.
    """
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.net = None
        if HAS_FACENET:
            self.net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            # Hook or truncate to get mid features? 
            # For simplicity, we use the final embedding as a high-level texture/id descriptor
            # Or better: "classify=False" gives embedding.
            
    def extract(self, images):
        if self.net is not None:
            images_resized = F.interpolate(images, size=(160, 160))
            emb = self.net(images_resized)
            return emb # (B, 512)
            
        # Mock
        return torch.randn(images.shape[0], 64, device=self.device)

class LocalFeatureExtractor(BaseFactorExtractor):
    """
    Extracts region-specific features (Eyes, Mouth).
    """
    def extract(self, images):
        B, C, H, W = images.shape
        # Improved crops
        eyes_crop = images[:, :, int(0.35*H):int(0.48*H), int(0.2*W):int(0.8*W)]
        mouth_crop = images[:, :, int(0.65*H):int(0.85*H), int(0.3*W):int(0.7*W)]
        
        eye_stats = torch.std(eyes_crop, dim=[2, 3]) # Texture variance in eyes
        mouth_stats = torch.std(mouth_crop, dim=[2, 3])
        
        return torch.cat([eye_stats, mouth_stats], dim=1)

class FactorWrapper:
    def __init__(self, device='cpu'):
        self.geom = GeometryExtractor(device)
        self.sym = SymmetryExtractor(device)
        self.tex = TextureExtractor(device)
        self.local = LocalFeatureExtractor(device)
    
    def extract_all(self, images):
        return {
            'geometry': self.geom.extract(images),
            'symmetry': self.sym.extract(images),
            'texture': self.tex.extract(images),
            'local': self.local.extract(images)
        }
