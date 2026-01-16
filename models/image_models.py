"""
Image classification models
Includes AlexNet and DenseNet for lung cancer detection
"""

import torch
import torch.nn as nn
from torchvision import models
from src.utils import disable_inplace_activations

def get_image_target_layer(model: nn.Module, model_name: str):
    """
    Get the best target layer for Grad-CAM for each image model.
    Supports:
    - torchvision AlexNet / DenseNet
    - TorchXRayVision DenseNet (CheX weights)
    """
    name = (model_name or "").lower()

    # -------- TorchXRayVision DenseNet121 (CheX / NIH / MIMIC) --------
    # Your UI name likely contains "xrv" or "chex"
    if ("xrv" in name) or ("chex" in name):
        # Best target: last conv in the last dense layer of denseblock4
        try:
            return model.features.denseblock4.denselayer16.conv2
        except Exception:
            # Fallback: last Conv2d in model.features
            target = None
            if hasattr(model, "features"):
                for m in model.features.modules():
                    if isinstance(m, nn.Conv2d):
                        target = m
            return target

    # -------- torchvision AlexNet --------
    if "alexnet" in name:
        return model.features[12]

    # -------- torchvision DenseNet --------
    if "densenet" in name:
        # safe target for torchvision DenseNet
        return model.features.denseblock4

    # -------- generic fallback --------
    target = None
    if hasattr(model, "features"):
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                target = m
    return target


def get_image_alexnet(num_classes: int = 2) -> nn.Module:
    """
    Get AlexNet adapted for lung cancer classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for Normal/Malignant)
        
    Returns:
        nn.Module: AlexNet model
    """
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = disable_inplace_activations(model)
    return model


def get_image_densenet(num_classes: int = 2) -> nn.Module:
    """
    Get DenseNet121 adapted for lung cancer classification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        nn.Module: DenseNet121 model
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = disable_inplace_activations(model)
    return model


def get_image_target_layer(model: nn.Module, model_name: str):
    """
    Get the best target layer for Grad-CAM for each image model.
    Deterministic + model-aware (recommended).
    """
    model_name = (model_name or "").lower()

    if "alexnet" in model_name:
        # AlexNet: last conv layer is model.features[12]
        return model.features[12]

    if "densenet" in model_name:
        # DenseNet: best target is the last dense block output
        # (more stable than "last Conv2d found")
        return model.features.denseblock4

    # Fallback: last Conv2d in features
    target = None
    if hasattr(model, "features"):
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                target = m
    return target
