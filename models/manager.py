"""
Model Manager
Handles model loading, caching, and device management
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.config import DEVICE

# -----------------------------
# PyTorch audio models
# -----------------------------
from models.audio_models import AudioCNN, get_audio_mobilenet, get_audio_vgg16

# -----------------------------
# PyTorch image models (optional baselines)
# -----------------------------
from models.image_models import get_image_alexnet, get_image_densenet


# -----------------------------
# Local model paths (your real files)
# -----------------------------
KERAS_AUDIO_PATH = os.path.join("models", "audio", "for", "audio_classifier.keras")
KERAS_AUDIO_LABELS = os.path.join("models", "audio", "for", "labels.json")
KERAS_AUDIO_CONFIG = os.path.join("models", "audio", "for", "config.json")

XRV_IMAGE_PTH = os.path.join("models", "image", "xrv-densenet121-res224-chex.pth")
XRV_PATHOLOGIES_TXT = os.path.join("models", "image", "xrv_pathologies.txt")


# -----------------------------
# Keras loaders (cached)
# -----------------------------
@lru_cache(maxsize=2)
def get_keras_audio_model(model_path: str = KERAS_AUDIO_PATH):
    """Load + cache your trained FoR Keras model (.keras)."""
    from tensorflow import keras

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    return keras.models.load_model(model_path)


@lru_cache(maxsize=1)
def get_keras_audio_metadata() -> Dict[str, Any]:
    """Load labels/config for your Keras audio model if present."""
    meta: Dict[str, Any] = {"labels": {0: "real", 1: "fake"}, "config": {}}

    if os.path.isfile(KERAS_AUDIO_LABELS):
        with open(KERAS_AUDIO_LABELS, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # normalize keys to int if possible
        try:
            meta["labels"] = {int(k): v for k, v in raw.items()}
        except Exception:
            meta["labels"] = raw

    if os.path.isfile(KERAS_AUDIO_CONFIG):
        with open(KERAS_AUDIO_CONFIG, "r", encoding="utf-8") as f:
            meta["config"] = json.load(f)

    return meta


# -----------------------------
# XRV loaders (cached)
# -----------------------------
@lru_cache(maxsize=1)
def get_xrv_image_model(
    pth_path: str = XRV_IMAGE_PTH,
    device: Union[str, torch.device] = DEVICE,
) -> nn.Module:
    """
    Load TorchXRayVision DenseNet121 + your local exported .pth weights.
    """
    import torchxrayvision as xrv

    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"XRV weights not found: {pth_path}")

    model = xrv.models.DenseNet(weights=None)  # prevents online download

    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@lru_cache(maxsize=1)
def get_xrv_pathologies(pathologies_path: str = XRV_PATHOLOGIES_TXT) -> Optional[list]:
    """Load XRV pathology labels saved to a txt file (one per line)."""
    if not os.path.isfile(pathologies_path):
        return None

    labels: list[str] = []
    with open(pathologies_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


# Backwards compatibility (your pipeline imported load_xrv_pathologies earlier)
def load_xrv_pathologies():
    return get_xrv_pathologies()


class ModelManager:
    """
    Manages PyTorch model loading + caching + device placement.
    Keras model loading is handled by get_keras_audio_model() above.
    """

    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.device = DEVICE

    # -----------------------------
    # PyTorch AUDIO models
    # -----------------------------
    def get_audio_model(self, model_name: str) -> nn.Module:
        """
        Get or load a PyTorch audio classification model.
        Accepts UI aliases like "Custom cnn" or "mobileNet".
        """
        name = model_name.strip().lower()
        key = f"audio::{name}"

        if key not in self.models:
            if name in ["custom cnn", "customcnn", "custom_cnn"]:
                model = AudioCNN(num_classes=2)

            elif name in ["mobilenet", "mobile net", "mobilenetv2", "mobilenet v2", "mobilenet (pytorch)", "mobilenet (baseline)"]:
                model = get_audio_mobilenet(num_classes=2)

            elif name in ["vgg16", "vgg-16", "vgg 16"]:
                model = get_audio_vgg16(num_classes=2)

            else:
                raise ValueError(f"Unknown PyTorch audio model: {model_name}")

            self.models[key] = model.to(self.device).eval()

        return self.models[key]

    # -----------------------------
    # IMAGE models (PyTorch baselines + XRV)
    # -----------------------------
    def get_image_model(self, model_name: str) -> nn.Module:
        """
        Get or load an image model.
        Supports:
          - AlexNet (baseline)
          - DenseNet (baseline)
          - XRV DenseNet121 (CheXpert) (local pth)
        """
        name = model_name.strip().lower()
        key = f"image::{name}"

        if key not in self.models:
            if name in ["alexnet", "alex net"]:
                model = get_image_alexnet(num_classes=2)
                self.models[key] = model.to(self.device).eval()

            elif name in ["densenet", "dense net"]:
                model = get_image_densenet(num_classes=2)
                self.models[key] = model.to(self.device).eval()

            elif name in [
                "xrv densenet121 (chexpert)",
                "xrv densenet121 chex",
                "xrv densenet121 chexpert",
                "xrv densenet121 (chex)",
                "torchxrayvision",
                "xrv",
                "xrv densenet121",
            ]:
                model = get_xrv_image_model()
                self.models[key] = model  # already on device + eval()

            else:
                raise ValueError(f"Unknown image model: {model_name}")

        return self.models[key]


# Global model manager instance
model_manager = ModelManager()

