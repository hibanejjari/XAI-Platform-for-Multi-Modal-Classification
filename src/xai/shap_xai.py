"""
SHAP (SHapley Additive exPlanations) - REGION MASKING VERSION
Produces blob/patch-like explanations (LIME-like logic) for audio spectrogram CNNs.

This is intentionally different from DeepExplainer/GradientExplainer:
- It masks regions (superpixels/patches) ON/OFF like your notebook logic.
- Uses shap.KernelExplainer over region masks.
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
    SHAP explanation using a masking-based approach (KernelExplainer) + region segmentation.
    Matches LIME-style logic from your notebook and yields blob-like explanations.

    Args:
        model: PyTorch model
        input_tensor: (B,C,H,W) spectrogram tensor

    Returns:
        (fig, explanation_text) or (None, error_message)
    """
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed. Install: pip install shap"

    try:
        from skimage.segmentation import slic
        from skimage.color import gray2rgb

        model.eval()
        device = input_tensor.device

        # ---- Predict class ----
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            pred_class = int(out.argmax(dim=1).item())
            num_classes = int(probs.shape[1])
            confidence = float(probs[0, pred_class].item())

        # ---- Prepare spectrogram for segmentation ----
        spec_chw = input_tensor[0].detach().cpu().numpy()  # (C,H,W)
        C, H, W = spec_chw.shape

        # Use channel 0 for segmentation/visualization
        spec_hw = spec_chw[0].astype(np.float32)

        # Normalize to [0,1] for stable segmentation
        spec_norm = (spec_hw - spec_hw.min()) / (spec_hw.max() - spec_hw.min() + 1e-8)
        img = gray2rgb(spec_norm)  # (H,W,3)

        # ---- Region segmentation (LIME-like) ----
        segments = slic(img, n_segments=40, compactness=10, sigma=1, start_label=0)
        n_regions = int(segments.max() + 1)

        # ---- Build masked tensor from region mask vector ----
        def build_masked_tensor(z01: np.ndarray):
            masked = spec_chw.copy()  # (C,H,W)
            keep = np.zeros((H, W), dtype=np.float32)

            # Keep regions where z01[r]=1
            for r in range(n_regions):
                if z01[r] > 0.5:
                    keep[segments == r] = 1.0

            # Apply keep mask to all channels
            for c in range(C):
                masked[c] = masked[c] * keep

            return torch.tensor(masked, dtype=input_tensor.dtype).unsqueeze(0).to(device)

        # ---- Prediction function for KernelExplainer ----
        def predict_fn(z):
            """
            z: (N, n_regions) mask vectors
            returns: (N, num_classes) probabilities
            """
            z = np.array(z, dtype=np.float32)
            preds = []

            with torch.no_grad():
                for i in range(z.shape[0]):
                    xt = build_masked_tensor(z[i])
                    o = model(xt)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    p = torch.softmax(o, dim=1).detach().cpu().numpy()
                    preds.append(p[0])

            return np.vstack(preds)

        # ---- SHAP KernelExplainer ----
        background = np.zeros((1, n_regions), dtype=np.float32)  # all regions OFF
        explainer = shap.KernelExplainer(predict_fn, background)

        x = np.ones((1, n_regions), dtype=np.float32)  # all regions ON

        shap_values = explainer.shap_values(x, nsamples=200)

        # ---- Extract SHAP vector for the predicted class ----
        # Possible shapes:
        #  - list[class] => (1, n_regions) or (1, n_regions, K)
        #  - array => (1, n_regions) or (1, n_regions, K)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[pred_class])[0]
        else:
            sv = np.array(shap_values)[0]

        # If sv is (n_regions, num_classes), pick pred_class column
        if sv.ndim == 2 and sv.shape[1] == num_classes:
            sv = sv[:, pred_class]

        sv = np.squeeze(sv)

        if sv.ndim != 1 or sv.shape[0] != n_regions:
            raise ValueError(f"Unexpected SHAP shape. Expected (n_regions,), got {sv.shape} with n_regions={n_regions}")

        # ---- Build region heatmap (H,W) ----
        shap_map = np.zeros((H, W), dtype=np.float32)
        for r in range(n_regions):
            shap_map[segments == r] = float(sv[r])

        # ---- Plot ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(spec_hw, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Input Spectrogram\nPredicted: Class {pred_class}")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")
        plt.colorbar(im0, ax=axes[0])

        vmax = np.percentile(np.abs(shap_map), 99)
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = float(np.abs(shap_map).max() + 1e-12)

        im1 = axes[1].imshow(shap_map, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        axes[1].set_title("SHAP (Region Masking)\n(Red=Positive, Green=Negative)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")
        plt.colorbar(im1, ax=axes[1])

        axes[2].imshow(spec_hw, aspect="auto", origin="lower", cmap="gray", alpha=0.65)
        im2 = axes[2].imshow(shap_map, aspect="auto", origin="lower", cmap="RdYlGn", vmin=-vmax, vmax=vmax, alpha=0.35)
        axes[2].set_title("Overlay (Region SHAP)")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Frequency")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()

        explanation = (
            f"SHAP (masking-based, LIME-like): predicted class {pred_class} with {confidence:.2%} confidence. "
            f"Computed over {n_regions} regions."
        )

        return fig, explanation

    except Exception as e:
        import traceback
        print(f"SHAP error: {e}\n{traceback.format_exc()}")
        return None, f"SHAP unavailable: {str(e)}"
