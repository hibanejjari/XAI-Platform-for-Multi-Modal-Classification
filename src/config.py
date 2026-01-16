"""
Configuration file for the Unified XAI Platform
Contains constants, class names, available models, and XAI compatibility mappings
"""

# -----------------------
# Classification labels
# -----------------------
AUDIO_CLASSES = ["Real", "Fake"]

# Baseline/binary image models only (AlexNet/DenseNet)
IMAGE_CLASSES = ["Normal", "Malignant"]


# -----------------------
# Available models (UI labels)
# -----------------------
# Audio models
AUDIO_MODELS = [
    "FoR Keras (local)",   # your trained TensorFlow/Keras model
    "Custom CNN",          # PyTorch baseline
    "MobileNet",           # PyTorch baseline
    "VGG16"                # PyTorch baseline
]

# Image models
IMAGE_MODELS = [
    "XRV DenseNet121 (CheXpert)",  # TorchXRayVision pretrained weights (.pth local)
    "AlexNet",                     # optional baseline
    "DenseNet"                     # optional baseline
]


# -----------------------
# Available XAI methods
# -----------------------
AUDIO_XAI_METHODS = ["Grad-CAM", "SHAP", "LIME"]
IMAGE_XAI_METHODS = ["Grad-CAM", "LIME"]


# -----------------------
# XAI compatibility mapping
# (drives filtering + prevents wrong combos)
# -----------------------
XAI_COMPATIBILITY = {
    "Audio": {
        # ✅ now enabled because you said you have TF Grad-CAM/LIME/SHAP
        "FoR Keras (local)": ["SHAP", "LIME"],

        # PyTorch audio models
        "Custom CNN": ["Grad-CAM", "SHAP", "LIME"],
        "VGG16": ["Grad-CAM", "SHAP", "LIME"],
    },
    "Image": {
        # XRV supports Grad-CAM + LIME (SHAP optional if you implemented it)
        "XRV DenseNet121 (CheXpert)": ["Grad-CAM", "LIME"],

        # Baseline image models
        "AlexNet": ["Grad-CAM", "LIME"],
        "DenseNet": ["Grad-CAM", "LIME"],
    }
}


# -----------------------
# Audio preprocessing parameters (must match FoR training)
# -----------------------
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 128
AUDIO_DURATION = 2
AUDIO_HOP_LENGTH = 512


# -----------------------
# Image preprocessing parameters
# -----------------------
IMAGE_TARGET_SIZE = (224, 224)

# If your image preprocessing produces grayscale 1-channel tensors:
IMAGE_NORMALIZE_MEAN = [0.5]
IMAGE_NORMALIZE_STD = [0.5]

# If you later switch to 3-channel images (RGB), use:
# IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
# IMAGE_NORMALIZE_STD  = [0.229, 0.224, 0.225]


# -----------------------
# Local model paths (centralized)
# -----------------------
import os

# Audio Keras model files (local)
AUDIO_KERAS_MODEL_PATH = os.path.join("models", "audio", "for", "audio_classifier.keras")
AUDIO_KERAS_LABELS_PATH = os.path.join("models", "audio", "for", "labels.json")
AUDIO_KERAS_CONFIG_PATH = os.path.join("models", "audio", "for", "config.json")

# Image XRV weights (local)
XRV_IMAGE_WEIGHTS_PATH = os.path.join("models", "image", "xrv-densenet121-res224-chex.pth")
XRV_PATHOLOGIES_PATH = os.path.join("models", "image", "xrv_pathologies.txt")


# -----------------------
# Device configuration (PyTorch only)
# -----------------------
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
