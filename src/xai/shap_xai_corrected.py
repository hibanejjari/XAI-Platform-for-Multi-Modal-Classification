"""
SHAP (SHapley Additive exPlanations) - CORRECTED VERSION
Fixed implementation for audio classification that actually works
Based on successful implementations from Deepfake-Audio-Detection repositories
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


def apply_shap_audio(model, input_tensor):
    """
    Apply SHAP explanation to an audio classification model.
    CORRECTED: Uses proper background data and handles tensor shapes correctly.
    
    Args:
        model: PyTorch classification model
        input_tensor: Input tensor (B, C, H, W) - spectrogram
        
    Returns:
        tuple: (matplotlib figure, explanation text) or (None, error message)
    """
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed. Install: pip install shap"

    try:
        model.eval()
        device = input_tensor.device
        
        # CRITICAL FIX 1: Create proper background dataset
        # Use multiple samples with slight noise variations instead of just repeating
        background_size = 16  # Reduced from 8 for better estimation
        background = input_tensor.detach().clone()
        
        # Add slight variations to create a meaningful background
        backgrounds = []
        for i in range(background_size):
            # Add small gaussian noise for variation
            noise_scale = 0.1
            noisy = background + torch.randn_like(background) * noise_scale
            backgrounds.append(noisy)
        
        background = torch.cat(backgrounds, dim=0).to(device)
        
        # CRITICAL FIX 2: Use DeepExplainer instead of GradientExplainer
        # DeepExplainer is more stable for CNNs
        try:
            explainer = shap.DeepExplainer(model, background)
        except:
            # Fallback to GradientExplainer if DeepExplainer fails
            explainer = shap.GradientExplainer(model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_tensor)
        
        # Get predicted class
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred_class = int(out.argmax(dim=1).item())
            probs = torch.softmax(out, dim=1)
        
        # CRITICAL FIX 3: Proper handling of SHAP values based on output type
        if isinstance(shap_values, list):
            # Multi-class output: shap_values is a list
            sv = shap_values[pred_class]  # Get values for predicted class
        else:
            # Binary output: shap_values is already the array we need
            sv = shap_values
        
        # Handle different tensor shapes
        if len(sv.shape) == 4:  # (B, C, H, W)
            sv_map = sv[0].mean(axis=0)  # Average over channels
        elif len(sv.shape) == 3:  # (B, H, W)
            sv_map = sv[0]
        else:
            sv_map = sv
        
        # Get input spectrogram for visualization
        input_spec = input_tensor[0, 0].detach().cpu().numpy()
        
        # CRITICAL FIX 4: Better visualization with proper colormap
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original spectrogram
        im1 = axes[0].imshow(input_spec, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Input Spectrogram\nPredicted: Class {pred_class}")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")
        plt.colorbar(im1, ax=axes[0])
        
        # SHAP values heatmap
        im2 = axes[1].imshow(sv_map, aspect="auto", origin="lower", cmap="RdBu_r", 
                            vmin=-np.abs(sv_map).max(), vmax=np.abs(sv_map).max())
        axes[1].set_title("SHAP Values\n(Red=Positive, Blue=Negative)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")
        plt.colorbar(im2, ax=axes[1])
        
        # Overlay: Original + SHAP
        axes[2].imshow(input_spec, aspect="auto", origin="lower", cmap="gray", alpha=0.6)
        im3 = axes[2].imshow(np.abs(sv_map), aspect="auto", origin="lower", 
                            cmap="hot", alpha=0.4)
        axes[2].set_title("Overlay\n(Bright = Important)")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Frequency")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Create explanatory text
        confidence = probs[0][pred_class].item()
        explanation = (f"SHAP Analysis: Predicted class {pred_class} with {confidence:.2%} confidence. "
                      f"Red regions in SHAP map increase the prediction, blue regions decrease it. "
                      f"Brighter areas in overlay indicate more important frequency-time regions.")
        
        return fig, explanation
        
    except Exception as e:
        import traceback
        error_msg = f"SHAP error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Debug output
        return None, f"SHAP unavailable: {str(e)}"


def apply_shap_audio_simple(model, input_tensor):
    """
    Simplified SHAP implementation using KernelExplainer as fallback.
    This is slower but more reliable.
    
    Args:
        model: PyTorch classification model
        input_tensor: Input tensor (B, C, H, W)
        
    Returns:
        tuple: (matplotlib figure, explanation text) or (None, error message)
    """
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed. Install: pip install shap"
    
    try:
        model.eval()
        device = input_tensor.device
        
        # Flatten input for KernelExplainer
        input_flat = input_tensor.flatten().cpu().numpy().reshape(1, -1)
        
        # Create prediction function
        def predict_fn(x):
            """Prediction function for SHAP KernelExplainer"""
            with torch.no_grad():
                # Reshape back to image format
                original_shape = input_tensor.shape
                x_tensor = torch.FloatTensor(x).reshape(original_shape).to(device)
                out = model(x_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs = torch.softmax(out, dim=1)
                return probs.cpu().numpy()
        
        # Use KernelExplainer (slower but more reliable)
        # Create background as zeros
        background = np.zeros((1, input_flat.shape[1]))
        
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(input_flat, nsamples=100)
        
        # Get predicted class
        pred_probs = predict_fn(input_flat)
        pred_class = int(pred_probs.argmax())
        
        # Reshape SHAP values back to image
        if isinstance(shap_values, list):
            sv = shap_values[pred_class]
        else:
            sv = shap_values
            
        sv_image = sv.reshape(input_tensor.shape[2], input_tensor.shape[3])
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        input_spec = input_tensor[0, 0].cpu().numpy()
        
        axes[0].imshow(input_spec, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Input Spectrogram")
        
        im = axes[1].imshow(np.abs(sv_image), aspect="auto", origin="lower", cmap="hot")
        axes[1].set_title("SHAP Importance")
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        return fig, "SHAP computed with KernelExplainer (simplified method)"
        
    except Exception as e:
        return None, f"SHAP simple method failed: {str(e)}"
