"""
Dual-stream fusion model: separate ConvNeXt backbones for RGB and polar,
features concatenated before a shared classification head.
"""

import timm
import torch
import torch.nn as nn


class PolariseFusion(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "convnext_base",
                 drop_rate: float = 0.2):
        super().__init__()

        # RGB stream — ImageNet pretrained
        self.rgb_backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )

        # Polar stream — randomly initialised, 5-channel input
        self.polar_backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0,
            global_pool="avg", in_chans=5
        )

        feature_dim = self.rgb_backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, polar: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(rgb)
        polar_feat = self.polar_backbone(polar)
        return self.head(torch.cat([rgb_feat, polar_feat], dim=1))


class RGBBaseline(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "convnext_base",
                 drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, rgb: torch.Tensor, polar: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.backbone(rgb))


class PolarBaseline(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "convnext_base",
                 drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0,
            global_pool="avg", in_chans=5
        )
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, rgb: torch.Tensor = None, polar: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.backbone(polar))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    models = {
        "rgb": RGBBaseline,
        "polar": PolarBaseline,
        "fusion": PolariseFusion,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models)}")
    return models[model_name](num_classes=num_classes)
