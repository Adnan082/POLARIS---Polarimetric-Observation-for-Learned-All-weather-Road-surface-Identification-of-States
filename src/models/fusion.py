"""
Dual-stream ConvNeXt-Tiny models for road surface classification.

Three modes:
  rgb    — RGB-only baseline (pretrained ConvNeXt-Tiny)
  polar  — Polar-only baseline (random-init ConvNeXt-Tiny, 6-channel input)
  fusion — Two streams + spatial attention + late fusion head
"""

import torch
import torch.nn as nn
import timm


BACKBONE = "convnext_tiny"
FEATURE_DIM = 768   # ConvNeXt-Tiny output after final stage
POLAR_CHANS = 6     # S0/S1/S2/DoLP/sin_AoLP/cos_AoLP


class SpatialAttention(nn.Module):
    """1×1 conv gate — suppresses sky, focuses on road region."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class StreamEncoder(nn.Module):
    """ConvNeXt-Tiny backbone + spatial attention + global average pool."""

    def __init__(self, in_chans: int, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",   # return feature map, not pooled vector
            in_chans=in_chans,
        )
        self.attention = SpatialAttention(FEATURE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)              # (B, 768, H', W')
        feat = self.attention(feat)          # attended feature map
        return feat.mean(dim=[-2, -1])       # GAP → (B, 768)


class RGBBaseline(nn.Module):
    def __init__(self, num_classes: int, drop_rate: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.encoder = StreamEncoder(in_chans=3, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(drop_rate),
            nn.Linear(FEATURE_DIM, num_classes),
        )

    def forward(self, rgb: torch.Tensor, polar: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.encoder(rgb))


class PolarBaseline(nn.Module):
    def __init__(self, num_classes: int, drop_rate: float = 0.2, pretrained: bool = False):
        super().__init__()
        self.encoder = StreamEncoder(in_chans=POLAR_CHANS, pretrained=False)  # always random init
        self.head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(drop_rate),
            nn.Linear(FEATURE_DIM, num_classes),
        )

    def forward(self, rgb: torch.Tensor = None, polar: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.encoder(polar))


class PolariseFusion(nn.Module):
    def __init__(self, num_classes: int, drop_rate: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.rgb_stream   = StreamEncoder(in_chans=3,           pretrained=pretrained)
        self.polar_stream = StreamEncoder(in_chans=POLAR_CHANS, pretrained=False)
        self.head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM * 2),
            nn.Linear(FEATURE_DIM * 2, 512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, polar: torch.Tensor) -> torch.Tensor:
        rgb_feat   = self.rgb_stream(rgb)
        polar_feat = self.polar_stream(polar)
        return self.head(torch.cat([rgb_feat, polar_feat], dim=1))

    def load_baseline_weights(self, rgb_ckpt: str, polar_ckpt: str):
        """Warm-start fusion from separately trained baselines."""
        rgb_state   = torch.load(rgb_ckpt,   map_location="cpu")["model"]
        polar_state = torch.load(polar_ckpt, map_location="cpu")["model"]
        self.rgb_stream.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in rgb_state.items() if "encoder" in k}
        )
        self.polar_stream.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in polar_state.items() if "encoder" in k}
        )


def build_model(mode: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    models = {
        "rgb":    RGBBaseline,
        "polar":  PolarBaseline,
        "fusion": PolariseFusion,
    }
    if mode not in models:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(models)}")
    return models[mode](num_classes=num_classes, pretrained=pretrained)
