"""
Audio classification pipeline - CORRECTED VERSION WITH KERAS XAI
- Supports PyTorch audio models (VGG16/MobileNet/Custom CNN)
- Supports local Keras FoR model: models/audio/for/audio_classifier.keras
- NOW INTEGRATES TensorFlow XAI methods for Keras model
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from src.config import AUDIO_CLASSES, DEVICE
from src.preprocessing.audio import preprocess_audio

from models.manager import model_manager, get_keras_audio_model
from models.audio_models import get_audio_target_layer

# PyTorch XAI
from src.xai.gradcam import GradCAM
from src.xai.lime_audio import apply_lime_audio
from src.xai.shap_xai import apply_shap_audio

# TensorFlow/Keras XAI
from src.xai.gradcam_tf import gradcam_keras
from src.xai.lime_audio_tf import lime_audio_spectrogram
from src.xai.shap_audio_tf import shap_audio_kernel

KERAS_AUDIO_PATH = os.path.join("models", "audio", "for", "audio_classifier.keras")


def apply_gradcam_audio(model, input_tensor, mel_spec_original, model_name: str):
    """
    Apply Grad-CAM to PyTorch audio classification models.
    """
    target_layer = get_audio_target_layer(model, model_name)
    if target_layer is None:
        return None, "Grad-CAM not available for this model"

    cam_engine = GradCAM(model, target_layer)
    cam = cam_engine.generate_cam(input_tensor)
    cam_engine.remove_hooks()

    # Resize CAM to match original spectrogram size (mel_spec_original is 128 x T)
    cam_resized = cv2.resize(cam, (mel_spec_original.shape[1], mel_spec_original.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original Mel-Spectrogram")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
    axes[2].imshow(cam_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay - Important Regions")
    axes[2].axis("off")

    plt.tight_layout()
    return fig, "Grad-CAM highlights the frequency-time regions most important for classification."


def apply_gradcam_audio_keras(model, x: np.ndarray, mel_spec_original: np.ndarray):
    """
    Apply Grad-CAM to Keras audio classification models.
    
    Args:
        model: Keras model
        x: input tensor shape (1, 128, T, 1)
        mel_spec_original: original mel-spectrogram (128, T) for visualization
        
    Returns:
        (figure, explanation_text)
    """
    try:
        # Generate Grad-CAM heatmap
        cam = gradcam_keras(model, x)  # returns (H, W) normalized [0,1]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original Mel-Spectrogram")
        axes[0].axis("off")
        
        axes[1].imshow(cam, aspect="auto", origin="lower", cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[2].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay - Important Regions")
        axes[2].axis("off")
        
        plt.tight_layout()
        return fig, "Grad-CAM highlights the frequency-time regions most important for classification."
        
    except Exception as e:
        import traceback
        error_msg = f"Grad-CAM failed: {str(e)}\n{traceback.format_exc()[:200]}"
        print(error_msg)
        return None, error_msg


def apply_lime_audio_keras(model, x: np.ndarray, mel_spec_original: np.ndarray):
    """
    Apply LIME to Keras audio classification models.
    
    Args:
        model: Keras model
        x: input tensor shape (1, 128, T, 1)
        mel_spec_original: original mel-spectrogram (128, T)
        
    Returns:
        (figure, explanation_text)
    """
    try:
        # Generate LIME heatmap
        heatmap = lime_audio_spectrogram(
            model, x, 
            n_samples=600,  # adjust for speed/quality tradeoff
            grid=(8, 8)
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original Mel-Spectrogram")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
        axes[1].set_title("LIME Importance Map")
        axes[1].axis("off")
        
        axes[2].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[2].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay - Important Regions")
        axes[2].axis("off")
        
        plt.tight_layout()
        return fig, "LIME shows which frequency-time regions contributed most to the prediction."
        
    except Exception as e:
        import traceback
        error_msg = f"LIME failed: {str(e)}\n{traceback.format_exc()[:200]}"
        print(error_msg)
        return None, error_msg


def apply_shap_audio_keras(model, x: np.ndarray, mel_spec_original: np.ndarray):
    """
    Apply SHAP to Keras audio classification models.
    
    Args:
        model: Keras model
        x: input tensor shape (1, 128, T, 1)
        mel_spec_original: original mel-spectrogram (128, T)
        
    Returns:
        (figure, explanation_text)
    """
    try:
        # Generate SHAP heatmap (this may take a while)
        heatmap = shap_audio_kernel(
            model, x,
            nsamples=200  # adjust for speed (lower = faster but less accurate)
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original Mel-Spectrogram")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
        axes[1].set_title("SHAP Values")
        axes[1].axis("off")
        
        axes[2].imshow(mel_spec_original, aspect="auto", origin="lower", cmap="viridis")
        axes[2].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay - Important Regions")
        axes[2].axis("off")
        
        plt.tight_layout()
        return fig, "SHAP values show pixel-level importance using game theory principles."
        
    except Exception as e:
        import traceback
        error_msg = f"SHAP failed: {str(e)}\n{traceback.format_exc()[:200]}"
        print(error_msg)
        return None, error_msg


def _format_probs_for_display(prob_real: float, prob_fake: float, pred_idx: int, model_name: str) -> str:
    pred_label = AUDIO_CLASSES[pred_idx] if pred_idx < len(AUDIO_CLASSES) else str(pred_idx)
    result = f"**Prediction:** {pred_label}\n"
    result += f"**Confidence:** Real: {prob_real:.2%}, Fake: {prob_fake:.2%}\n"
    result += f"**Model:** {model_name}"
    return result


def classify_audio(audio_file: str, model_name: str, xai_method: str):
    """
    Complete audio classification pipeline with XAI.

    Returns:
        (result_text, visualization_figure, explanation_text)
    """
    try:
        # 1) Preprocess audio to mel-spectrogram
        # preprocess_audio returns (128, T) float32 normalized [0,1]
        mel_spec = preprocess_audio(audio_file)

        # =========================================================
        # 2) KERAS BRANCH (your trained FoR model)
        # =========================================================
        if ("keras" in model_name.lower()) or ("for" in model_name.lower()):
            keras_model = get_keras_audio_model(KERAS_AUDIO_PATH)

            # Keras expects (1, 128, T, 1)
            x = mel_spec[..., None]               # (128, T, 1)
            x = np.expand_dims(x, axis=0)         # (1, 128, T, 1)

            y = keras_model.predict(x, verbose=0)
            # y shape usually (1,1) with sigmoid
            prob_fake = float(y[0][0])
            prob_real = 1.0 - prob_fake
            pred_idx = 1 if prob_fake >= 0.5 else 0  # 0=Real, 1=Fake

            result = _format_probs_for_display(prob_real, prob_fake, pred_idx, model_name)

            # Apply XAI method (Keras/TensorFlow)
            if xai_method == "Grad-CAM":
                fig, txt = apply_gradcam_audio_keras(keras_model, x, mel_spec)
                return result, fig, txt

            elif xai_method == "SHAP":
                fig, txt = apply_shap_audio_keras(keras_model, x, mel_spec)
                return result, fig, txt

            elif xai_method == "LIME":
                fig, txt = apply_lime_audio_keras(keras_model, x, mel_spec)
                return result, fig, txt

            else:
                return result, None, "No XAI selected."

        # =========================================================
        # 3) PYTORCH BRANCH (existing repo models)
        # =========================================================

        # Most repo CNNs expect (128,128). Your mel is (128,T~63). Resize for torch models.
        mel_for_torch = cv2.resize(mel_spec, (128, 128), interpolation=cv2.INTER_AREA)

        # Prepare input tensor [1, 1, 128, 128]
        x = torch.FloatTensor(mel_for_torch).unsqueeze(0).unsqueeze(0)

        # VGG/MobileNet expect 3 channels
        if model_name in ["MobileNet", "VGG16"]:
            x = x.repeat(1, 3, 1, 1)

        input_tensor = x.to(DEVICE)

        # Load model
        model = model_manager.get_audio_model(model_name)
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
            pred_class = int(np.argmax(probs))

        # Assumes class order [Real, Fake]
        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        result = _format_probs_for_display(prob_real, prob_fake, pred_class, model_name)

        # Apply XAI method (PyTorch only)
        if xai_method == "Grad-CAM":
            fig, txt = apply_gradcam_audio(model, input_tensor, mel_spec, model_name)
            return result, fig, txt

        elif xai_method == "SHAP":
            fig, txt = apply_shap_audio(model, input_tensor)
            return result, fig, txt

        elif xai_method == "LIME":
            fig, txt = apply_lime_audio(model, input_tensor)
            return result, fig, txt

        else:
            return result, None, f"XAI method '{xai_method}' not supported."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in classify_audio: {error_details}")
        return f"Error: {str(e)}", None, f"Error details: {error_details[:400]}"







