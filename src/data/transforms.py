"""Albumentations pipelines for RGB and polar channels."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_rgb_transforms(image_size: int, train: bool):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_polar_transforms(image_size: int, train: bool):
    # Per-channel stats from 4 EDA sessions (dry/snow/damp/wet), 6-channel order:
    # [S0, S1, S2, DoLP, sin(2·AoLP), cos(2·AoLP)]
    # cos_AoLP stats are estimates — recompute from full .npy dataset after precompute
    polar_mean = [186.758690, -17.185258,  0.934233, 0.160030,  0.164714,  0.0]
    polar_std  = [139.619924,  17.679148, 14.583693, 0.096059,  1.095443,  0.7]

    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=polar_mean, std=polar_std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=polar_mean, std=polar_std),
        ToTensorV2(),
    ])
