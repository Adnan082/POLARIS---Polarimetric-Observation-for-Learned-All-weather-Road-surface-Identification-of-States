"""Albumentations pipelines for RGB and polar channels."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

# 6-channel order: [S0, S1, S2, DoLP, sin(2·AoLP), cos(2·AoLP)]
# Stats from 4 EDA sessions (dry/snow/damp/wet)
# cos_AoLP (index 5) is estimated — recompute from full .npy dataset after precompute
POLAR_MEAN = [186.758690, -17.185258,  0.934233, 0.160030,  0.164714,  0.0]
POLAR_STD  = [139.619924,  17.679148, 14.583693, 0.096059,  1.095443,  0.7]


def get_spatial_transforms(image_size: int, train: bool) -> A.Compose:
    """
    Spatial augmentations applied JOINTLY to RGB and polar so both
    receive the exact same crop, flip, and resize.

    Usage:
        result     = transform(image=rgb_hwc, polar=polar_hwc)
        rgb_out    = result["image"]
        polar_out  = result["polar"]
    """
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
        ], additional_targets={"polar": "image"})
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
    ], additional_targets={"polar": "image"})


def get_rgb_normalize(train: bool) -> A.Compose:
    """RGB-specific: colour jitter (train only) + ImageNet normalisation."""
    augs = []
    if train:
        augs.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5))
        augs.append(A.GaussianBlur(p=0.2))
    augs += [A.Normalize(mean=RGB_MEAN, std=RGB_STD), ToTensorV2()]
    return A.Compose(augs)


def get_polar_normalize() -> A.Compose:
    """Polar-specific: per-channel normalisation, no colour augmentation."""
    return A.Compose([
        A.Normalize(mean=POLAR_MEAN, std=POLAR_STD),
        ToTensorV2(),
    ])


def get_rgb_transforms(image_size: int, train: bool) -> A.Compose:
    """Legacy single-stream RGB transform (kept for backward compat)."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ToTensorV2(),
    ])


def get_polar_transforms(image_size: int, train: bool) -> A.Compose:
    """Legacy single-stream polar transform (kept for backward compat)."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=POLAR_MEAN, std=POLAR_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=POLAR_MEAN, std=POLAR_STD),
        ToTensorV2(),
    ])
