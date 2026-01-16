"""
Image preprocessing module
Handles preprocessing of chest X-ray images (TorchXRayVision expects 1 channel)
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from src.config import IMAGE_TARGET_SIZE


def preprocess_image(image_path: str, target_size: tuple = IMAGE_TARGET_SIZE) -> torch.Tensor:
    """
    Preprocess chest X-ray image to tensor.

    Returns:
        torch.Tensor: Preprocessed image tensor (1, H, W)
    """
    img = Image.open(image_path).convert("L")  # grayscale (1 channel)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    return transform(img)
