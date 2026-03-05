"""
Augmentation factory functions for fundus (RGB) and OCT (grayscale) images.

Usage:
    tf = get_fundus_transforms(train=True, image_size=224)
    tf = get_oct_transforms(train=False, image_size=224)
"""

from torchvision import transforms


def get_fundus_transforms(train: bool, image_size: int) -> transforms.Compose:
    """
    Fundus transforms.
    Input:  PIL Image in RGB mode.
    Output: FloatTensor (3, image_size, image_size), ImageNet-normalised.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_oct_transforms(train: bool, image_size: int) -> transforms.Compose:
    """
    OCT transforms.
    Input:  PIL Image in grayscale ("L") mode.
    Output: FloatTensor (1, image_size, image_size), normalised to mean=0.5, std=0.5.
    """
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
