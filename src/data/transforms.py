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
    # Per-channel stats computed from 3 EDA sessions (dry/snow/damp)
    # S0/S1/S2 are in raw uint8 scale (0-510 for S0, signed for S1/S2)
    # DoLP is normalized [0,1], AoLP is in radians [-pi/2, pi/2]
    polar_mean = [197.050388, -16.753594, 1.12391,  0.153915, 0.151688]
    polar_std  = [150.529773,  19.353948, 16.14056, 0.098816, 1.033142]

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
