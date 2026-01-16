"""
Image classification pipeline - FULLY FIXED VERSION
All issues resolved:
- Syntax error fixed
- LIME import corrected
- XRV label index error fixed
- Grad-CAM visualization improved
"""

from __future__ import annotations

import traceback
from typing import Tuple, Optional

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

XRV_PATHOLOGIES_TXT = os.path.join("models", "image", "xrv_pathologies.txt")


from src.config import DEVICE, IMAGE_CLASSES
from src.preprocessing.image import preprocess_image

from models.manager import model_manager, get_xrv_pathologies

# Optional XAI imports (safe)
try:
    from src.xai.gradcam import GradCAM
except Exception:
    GradCAM = None

try:
    from src.xai.lime_xai import apply_lime_image as lime_xai_apply
    LIME_AVAILABLE = True
except Exception:
    lime_xai_apply = None
    LIME_AVAILABLE = False

# Optional: if you have a helper for picking target conv layers
try:
    from models.image_models import get_image_target_layer
except Exception:
    get_image_target_layer = None


def _is_xrv_model_name(model_name: str) -> bool:
    n = model_name.strip().lower()
    return n in {
        "xrv densenet121 (chexpert)",
        "xrv densenet121 (chex)",
        "xrv densenet121 chexpert",
        "xrv densenet121 chex",
        "xrv densenet121",
        "torchxrayvision",
        "xrv",
    }


def _maybe_repeat_channels_for_torchvision(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Torchvision AlexNet/DenseNet expect 3 channels. If we got grayscale (1 channel),
    repeat it to 3 channels.
    """
    if x.dim() != 4:
        return x

    if x.shape[1] != 1:
        return x

    # Try to inspect first conv
    first_conv = None
    try:
        if hasattr(model, "features"):
            # DenseNet: model.features.conv0
            if hasattr(model.features, "conv0"):
                first_conv = model.features.conv0
            # AlexNet: model.features[0]
            elif isinstance(model.features, torch.nn.Sequential) and len(model.features) > 0:
                first_conv = model.features[0]
    except Exception:
        first_conv = None

    # If we detected first conv expects 3 channels, repeat
    if first_conv is not None and hasattr(first_conv, "weight"):
        expected_c = int(first_conv.weight.shape[1])
        if expected_c == 3 and x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)

    # Fallback heuristic: AlexNet/DenseNet almost always want 3 channels
    return x.repeat(1, 3, 1, 1)


def _format_xrv_output(probs: torch.Tensor, top_k: int = 5) -> str:
    """
    Format XRV multi-label pathology predictions.
    FIXED: Handles label/output size mismatch gracefully.
    
    Args:
        probs: (num_labels,) sigmoid probabilities
        top_k: number of top predictions to show
        
    Returns:
        Formatted string with top predictions
    """
    labels = get_xrv_pathologies() or []
    num_outputs = int(probs.numel())
    
    # CRITICAL FIX: Handle label/output size mismatch
    if not labels or len(labels) != num_outputs:
        print(f"WARNING: Labels ({len(labels)}) don't match outputs ({num_outputs})")
        # Generate generic labels
        labels = [f"Pathology_{i}" for i in range(num_outputs)]

    k = min(top_k, num_outputs)
    vals, idx = torch.topk(probs, k=k)

    lines = ["**Top predicted pathologies (XRV CheXpert):**"]
    for v, i in zip(vals.tolist(), idx.tolist()):
        # Safe indexing with bounds check
        if i < len(labels):
            lines.append(f"- {labels[i]}: {v:.2%}")
        else:
            lines.append(f"- Pathology_{i}: {v:.2%}")
    
    return "\n".join(lines)


def _apply_gradcam_image(model: torch.nn.Module, input_tensor: torch.Tensor, 
                         model_name: str, original_image_path: str):
    """
    Apply Grad-CAM to image classification models.
    Returns matplotlib figure with 3-panel visualization.
    
    Returns: (fig, explanation_text) or (None, reason)
    """
    if GradCAM is None:
        return None, "Grad-CAM module not available (src.xai.gradcam import failed)."

    target_layer = None
    if get_image_target_layer is not None:
        try:
            target_layer = get_image_target_layer(model, model_name)
        except Exception:
            target_layer = None

    # Fallback: try common places
    if target_layer is None:
        # DenseNet
        if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            target_layer = model.features.denseblock4
        # AlexNet
        elif hasattr(model, "features") and isinstance(model.features, torch.nn.Sequential):
            # last conv-ish layer (best effort)
            for layer in reversed(list(model.features)):
                if isinstance(layer, torch.nn.Conv2d):
                    target_layer = layer
                    break

    if target_layer is None:
        return None, "Grad-CAM target layer not found for this model."

    try:
        # Generate Grad-CAM heatmap
        cam_engine = GradCAM(model, target_layer)
        cam = cam_engine.generate_cam(input_tensor)  # (H, W) numpy array [0,1]
        cam_engine.remove_hooks()

        # Load original image for visualization
        original_img = Image.open(original_image_path).convert('RGB')
        original_img = np.array(original_img)
        
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Create visualization with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
        axes[0].set_title("Original X-Ray Image")
        axes[0].axis("off")
        
        # Grad-CAM heatmap
        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # Overlay
        axes[2].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
        axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay - Important Regions")
        axes[2].axis("off")
        
        plt.tight_layout()
        return fig, "Grad-CAM highlights the image regions most important for the prediction."
        
    except Exception as e:
        import traceback
        error_msg = f"Grad-CAM failed: {str(e)}\n{traceback.format_exc()[:200]}"
        print(error_msg)
        return None, error_msg


def _apply_lime_image(model: torch.nn.Module, original_image_path: str):
    """
    Apply LIME to image classification models.
    Wrapper that properly calls the LIME implementation.
    
    Returns: (fig, explanation_text) or (None, reason)
    """
    if not LIME_AVAILABLE or lime_xai_apply is None:
        return None, "LIME not available. Install: pip install lime scikit-image"
    
    try:
        # Load original image as PIL
        original_img = Image.open(original_image_path).convert('RGB')
        
        # Call LIME with correct signature
        fig, txt = lime_xai_apply(model, original_img, DEVICE)
        return fig, txt
        
    except Exception as e:
        import traceback
        error_msg = f"LIME failed: {str(e)}\n{traceback.format_exc()[:200]}"
        print(error_msg)
        return None, error_msg

def get_xrv_pathologies(pathologies_path: str = XRV_PATHOLOGIES_TXT):
    if not os.path.isfile(pathologies_path):
        return None
    labels = []
    with open(pathologies_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def classify_image(image_file: str, model_name: str, xai_method: str):
    """
    Complete image classification pipeline with XAI.

    Returns:
      (result_markdown, visualization, explanation_text)

    visualization can be:
      - matplotlib figure (Grad-CAM, LIME)
      - None
    """
    try:
        # 1) preprocess image -> tensor expected shape (1, C, 224, 224)
        img_tensor = preprocess_image(image_file)  # should already return torch Tensor
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor)

        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        input_tensor = img_tensor.to(DEVICE)

        # 2) load model
        model = model_manager.get_image_model(model_name)

        # 3) forward pass
        with torch.no_grad():
            # For AlexNet/DenseNet baseline, convert grayscale->RGB
            if not _is_xrv_model_name(model_name):
                input_tensor = _maybe_repeat_channels_for_torchvision(model, input_tensor)

            logits = model(input_tensor)

        # 4) format prediction
        if _is_xrv_model_name(model_name):
            # XRV is multi-label -> sigmoid
            probs = torch.sigmoid(logits).squeeze(0)  # (num_labels,)
            result = _format_xrv_output(probs, top_k=5)
        else:
            # baseline binary -> softmax
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            result = f"**Prediction:** {IMAGE_CLASSES[pred]}\n"
            if probs.shape[1] >= 2:
                result += f"**Confidence:** Normal: {probs[0,0]:.2%}, Malignant: {probs[0,1]:.2%}\n"
            result += f"**Model:** {model_name}"

        # 5) XAI
        if xai_method == "Grad-CAM":
            fig, txt = _apply_gradcam_image(model, input_tensor, model_name, image_file)
            return result, fig, txt

        if xai_method == "LIME":
            fig, txt = _apply_lime_image(model, image_file)
            return result, fig, txt

        # No XAI selected
        if xai_method in (None, "", "None"):
            return result, None, "No XAI method selected."

        return result, None, f"XAI method '{xai_method}' not supported for images."

    except Exception as e:
        err = traceback.format_exc()
        print("Error in classify_image:\n", err)
        return f"Error: {str(e)}", None, err[:400]