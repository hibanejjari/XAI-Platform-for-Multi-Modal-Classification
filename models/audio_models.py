"""
Audio classification models
Includes Custom CNN and pretrained model wrappers (MobileNet, VGG16)
"""

import torch
import torch.nn as nn
from torchvision import models
from src.utils import disable_inplace_activations


class AudioCNN(nn.Module):
    """Custom CNN for Audio Classification (1-channel mel-spectrogram)"""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize Audio CNN.
        
        Args:
            num_classes: Number of output classes (default: 2 for Real/Fake)
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


def get_audio_mobilenet(num_classes: int = 2) -> nn.Module:
    """
    Get MobileNetV2 adapted for audio classification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        nn.Module: MobileNetV2 model
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = disable_inplace_activations(model)
    return model


def get_audio_vgg16(num_classes: int = 2) -> nn.Module:
    """
    Get VGG16 adapted for audio classification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        nn.Module: VGG16 model
    """
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = disable_inplace_activations(model)
    return model


def get_audio_target_layer(model: nn.Module, model_name: str):
    """
    Get the target layer for Grad-CAM visualization.
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        
    Returns:
        nn.Module: Target layer for Grad-CAM
    """
    if model_name == "Custom CNN":
        return model.conv_layers[6]  # Conv2d(64->128)
    elif model_name == "VGG16":
        return model.features[28]    # Last Conv2d in VGG16 features
    elif model_name == "MobileNet":
        return model.features[-1]    # Last feature block
    return None
