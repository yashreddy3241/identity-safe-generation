import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.fftpack

class SRMConv(nn.Module):
    """
    Steganalysis Rich Model (SRM) Filters.
    Extracts high-frequency noise residuals (artifacts) from images.
    This is critical for detecting GAN upsampling traces.
    """
    def __init__(self):
        super(SRMConv, self).__init__()
        self.w = self._get_srm_filters()
        
    def _get_srm_filters(self):
        # 3 Fixed SRM filters: Define the noise residuals
        # Filter 1: KV
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.array(filter1, dtype=float) / 4.0

        # Filter 2: Residual 3x3
        filter2 = [[-1, 2, -1],
                   [2, -4, 2],
                   [-1, 2, -1]]
        filter2 = np.array(filter2, dtype=float) / 4.0
        # Pad to 5x5
        filter2 = np.pad(filter2, ((1,1),(1,1)), 'constant')

        # Filter 3: High Pass
        filter3 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = np.array(filter3, dtype=float) / 12.0

        filters = np.stack([filter1, filter2, filter3]) # (3, 5, 5)
        filters = torch.from_numpy(filters).float().unsqueeze(1) # (3, 1, 5, 5)
        return nn.Parameter(filters, requires_grad=False)

    def forward(self, x):
        """
        Input: (B, 3, H, W) RGB Images
        Output: (B, 3, H, W) SRM Features (Gray-like residuals)
        """
        # Convert RGB to Grayscale for filter application
        # Standard weights: 0.299 R + 0.587 G + 0.114 B
        gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        
        # Apply filters
        # Result: (B, 3, H, W) (Since we have 3 filters)
        return F.conv2d(gray, self.w, padding=2)

class DCTTransform:
    """
    Discrete Cosine Transform (DCT) log-scale spectrum.
    Captures periodic grid artifacts from GANs.
    """
    def __call__(self, img_array):
        # img_array: numpy (H, W, 3) in [0, 255]
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        # 2D DCT
        dct = scipy.fftpack.dct(scipy.fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
        
        # Log scale
        dct_log = np.log(np.abs(dct) + 1e-12)
        
        # Resize/Crop to manageable size if needed (e.g. mostly low freq)
        # But we pass full spectrum for CNN processing.
        # Normalize to 0-1 range roughly for NN stability
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-6)
        
        # Add channel dim: (H, W, 1)
        return dct_norm[..., np.newaxis]
