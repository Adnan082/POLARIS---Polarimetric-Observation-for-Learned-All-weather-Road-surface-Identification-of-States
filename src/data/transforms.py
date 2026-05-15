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
    # Polar channels have no ImageNet statistics — normalize per-channel from data
    # TODO: compute mean/std from EDA and replace these placeholders
    polar_mean = [0.5] * 5
    polar_std = [0.25] * 5

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
