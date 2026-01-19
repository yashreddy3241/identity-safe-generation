import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.transforms import SRMConv

class RGBStream(nn.Module):
    """
    Stream 1: Spatial Analysis
    Backbone: EfficientNetV2-S (Pretrained) - Fast & Accurate.
    """
    def __init__(self, backbone_name='tf_efficientnetv2_s'):
        super(RGBStream, self).__init__()
        # Load from TIMM, remove classifier
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        # EfficientNetV2-S features: [24, 48, 64, 160, 256] channels at different scales
        self.out_channels = [24, 48, 64, 160, 256]

    def forward(self, x):
        # Returns list of features at 5 scales
        return self.backbone(x)

class FreqStream(nn.Module):
    """
    Stream 2: Frequency Analysis
    Input: (B, 3, H, W) DCT or SRM Filtered Images.
    Backbone: Custom or Lightweight ResNet.
    """
    def __init__(self):
        super(FreqStream, self).__init__()
        # SRM Filter layer (Fixed)
        self.srm = SRMConv()
        
        # Lightweight Feature Extractor (ResNet18-like)
        # Input 3 channels (SRM residuals)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Simple layers for prototype speed (can swap with full ResNet18)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is RGB image. SRM is applied internally or pre-computed.
        # But here we assume Model takes raw image and applies SRM internally for end-to-end.
        # However, dataset returns DCT/SRM tensors.
        # Let's support both. If input is 3 ch, we assume it's raw and apply SRM?
        # Actually, let's Stick to the plan: Dataset provides 'dct', but here 
        # let's assume this stream takes RGB and computes SRM residuals itself
        # to ensure gradients can flow if we wanted (though SRM is fixed).
        
        x_srm = self.srm(x) # (B, 3, H, W) residuals
        
        c1 = self.conv1(x_srm) # /4
        c2 = self.layer1(c1)   # /4
        c3 = self.layer2(c2)   # /8
        c4 = self.layer3(c3)   # /16
        c5 = self.layer4(c4)   # /32
        
        return [c1, c2, c3, c4, c5]

class CrossModalityFusion(nn.Module):
    """
    Fuses RGB and Frequency Features using Channel Attention.
    """
    def __init__(self, in_channels):
        super(CrossModalityFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, freq):
        # Resize freq to match rgb (if needed, usually matched by design)
        if rgb.shape != freq.shape:
            freq = F.interpolate(freq, size=rgb.shape[2:], mode='bilinear', align_corners=False)
            
        cat = torch.cat([rgb, freq], dim=1)
        fused = self.conv(cat)
        w = self.att(fused)
        return fused * w

class FreqDetectNet(nn.Module):
    """
    The Full SOTA Architecture.
    Dual Stream -> Fusion -> Decoder (Segmentation) + GAP (Classification)
    """
    def __init__(self, num_classes=1):
        super(FreqDetectNet, self).__init__()
        self.rgb_stream = RGBStream()
        self.freq_stream = FreqStream()
        
        # Fusion Layers for top 3 scales (deepest features)
        # EfficientNetV2-S: 64, 160, 256
        # ResNet Custom:   128, 256, 512
        # We project freq to match rgb dimensions
        self.proj_c3 = nn.Conv2d(128, 64, 1)
        self.proj_c4 = nn.Conv2d(256, 160, 1)
        self.proj_c5 = nn.Conv2d(512, 256, 1)
        
        self.fusion3 = CrossModalityFusion(64)
        self.fusion4 = CrossModalityFusion(160)
        self.fusion5 = CrossModalityFusion(256)
        
        # Decoder (Simple U-Net like upsampling)
        # Input: 256 channels (fusion5)
        self.up1 = self._up_block(256, 160)
        self.up2 = self._up_block(160, 64)
        self.up3 = self._up_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, 1) # Mask Output
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        
        # 1. Dual Streams
        rgb_feats = self.rgb_stream(x) # [c1, c2, c3, c4, c5]
        freq_feats = self.freq_stream(x) # [c1, ... c5]
        
        # 2. Fusion (Deepest 3 layers)
        # rgb[2] is 64ch, freq[2] is 128ch
        f3 = self.fusion3(rgb_feats[2], self.proj_c3(freq_feats[2]))
        
        # rgb[3] is 160ch, freq[3] is 256ch
        f4 = self.fusion4(rgb_feats[3], self.proj_c4(freq_feats[3]))
        
        # rgb[4] is 256ch, freq[4] is 512ch
        f5 = self.fusion5(rgb_feats[4], self.proj_c5(freq_feats[4]))
        
        # 3. Task 1: Classification (Real vs Fake)
        gap = self.global_pool(f5).flatten(1)
        cls_logits = self.classifier(gap)
        
        # 4. Task 2: Segmentation (Localization)
        u1 = self.up1(f5) + f4 # Skip connection approach
        u2 = self.up2(u1) + f3
        u3 = self.up3(u2)
        # Upsample to full resolution (from /8 to /1) -> x8
        mask_logits = self.final_conv(u3)
        mask_logits = F.interpolate(mask_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return cls_logits, mask_logits
