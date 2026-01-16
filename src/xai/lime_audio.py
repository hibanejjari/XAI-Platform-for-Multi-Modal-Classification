"""
LIME for AUDIO spectrograms (region-masking version)
- Treats mel-spectrogram as an "image"
- Segments into regions (SLIC)
- LIME perturbs regions ON/OFF, then fits a local surrogate
- Builds a region heatmap (blob style) like your SHAP masking visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    from skimage.segmentation import slic
    from skimage.color import gray2rgb
    LIME_AVAILABLE = True
except Exception:
    lime_image = None
    slic = None
    gray2rgb = None
    LIME_AVAILABLE = False


def apply_lime_audio(
    model,
    input_tensor: torch.Tensor,
    num_samples: int = 250,
    n_segments: int = 40,
    compactness: float = 10.0,
):
    """
    Apply LIME explanation to an audio CNN using spectrogram region masking.

    Args:
        model: torch model
        input_tensor: (B,C,H,W) spectrogram tensor
        num_samples: LIME perturbations (higher = better, slower)
        n_segments: number of SLIC regions (higher = smaller blobs)
        compactness: SLIC compactness

    Returns:
        (fig, text) or (None, error_msg)
    """
    if not LIME_AVAILABLE:
        return None, "LIME not installed. Install: pip install lime scikit-image"

    try:
        model.eval()
        device = input_tensor.device

        # ---- Prediction (get pred class + confidence) ----
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            pred_class = int(out.argmax(dim=1).item())
            num_classes = int(probs.shape[1])
            confidence = float(probs[0, pred_class].item())

        # ---- Prepare spectrogram image for LIME (H,W,3 uint8) ----
        # input_tensor: (1,C,H,W)
        x_np = input_tensor[0].detach().cpu().numpy()  # (C,H,W)
        C, H, W = x_np.shape

        # Use channel 0 as the "display" spectrogram
        spec_hw = x_np[0].astype(np.float32)

        # Keep original scale so we can map perturbed images back correctly
        spec_min = float(spec_hw.min())
        spec_max = float(spec_hw.max())
        denom = (spec_max - spec_min) + 1e-8

        # Normalize to [0..1] for segmentation, then to uint8 for LIME
        spec_norm01 = (spec_hw - spec_min) / denom
        img01 = gray2rgb(spec_norm01)  # (H,W,3) float [0..1]
        img_uint8 = (img01 * 255.0).clip(0, 255).astype(np.uint8)

        # ---- Custom segmentation function (SLIC) ----
        # LIME will call this once to get segments for the original image
        def segmentation_fn(image_uint8):
            image01 = image_uint8.astype(np.float32) / 255.0
            return slic(
                image01,
                n_segments=n_segments,
                compactness=compactness,
                sigma=1,
                start_label=0,
            )

        # ---- Prediction function for LIME ----
        # LIME passes a list/array of perturbed images (H,W,3) uint8
        def predict_fn(images_uint8):
            images_uint8 = np.array(images_uint8)
            batch_probs = []

            with torch.no_grad():
                for im in images_uint8:
                    # im: (H,W,3) uint8 -> back to [0..1]
                    im01 = im.astype(np.float32) / 255.0

                    # convert to grayscale (we only used 1 ch originally)
                    # since im is gray2rgb, just take channel 0
                    pert_norm01 = im01[..., 0]

                    # map back to original spec scale
                    pert_spec = (pert_norm01 * denom + spec_min).astype(np.float32)  # (H,W)

                    # Build tensor shape to match model input (C,H,W)
                    if C == 1:
                        tens = torch.tensor(pert_spec, dtype=input_tensor.dtype).unsqueeze(0)  # (1,H,W)
                    else:
                        # if model expects 3 channels, replicate pert_spec
                        tens = torch.tensor(pert_spec, dtype=input_tensor.dtype).unsqueeze(0).repeat(C, 1, 1)

                    tens = tens.unsqueeze(0).to(device)  # (1,C,H,W)

                    o = model(tens)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    p = torch.softmax(o, dim=1).detach().cpu().numpy()[0]
                    batch_probs.append(p)

            return np.vstack(batch_probs)

        # ---- LIME Explainer ----
        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            img_uint8,
            predict_fn,
            top_labels=min(2, num_classes),
            hide_color=0,              # masked regions -> black
            num_samples=num_samples,
            segmentation_fn=segmentation_fn,
        )

        # ---- Build region-weight map (blob style) ----
        segments = explanation.segments  # (H,W) with region ids
        n_regions = int(segments.max() + 1)

        # local_exp[label] is list of (region_id, weight)
        weights = np.zeros(n_regions, dtype=np.float32)
        for rid, w in explanation.local_exp[pred_class]:
            if 0 <= rid < n_regions:
                weights[rid] = float(w)

        lime_map = np.zeros((H, W), dtype=np.float32)
        for r in range(n_regions):
            lime_map[segments == r] = weights[r]

        # ---- Plot (match SHAP style: original / map / overlay) ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(spec_hw, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Input Spectrogram\nPredicted: Class {pred_class}")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")
        plt.colorbar(im0, ax=axes[0])

        # percentile scaling for contrast
        vmax = np.percentile(np.abs(lime_map), 99)
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = float(np.abs(lime_map).max() + 1e-12)

        im1 = axes[1].imshow(lime_map, aspect="auto", origin="lower",
                             cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        axes[1].set_title("LIME (Region Masking)\n(Red=Positive, Green=Negative)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")
        plt.colorbar(im1, ax=axes[1])

        axes[2].imshow(spec_hw, aspect="auto", origin="lower", cmap="gray", alpha=0.65)
        im2 = axes[2].imshow(lime_map, aspect="auto", origin="lower",
                             cmap="RdYlGn", vmin=-vmax, vmax=vmax, alpha=0.35)
        axes[2].set_title("Overlay (Region LIME)")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Frequency")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()

        txt = (
            f"LIME (region-masking): predicted class {pred_class} with {confidence:.2%} confidence. "
            f"Computed over ~{n_regions} regions. "
            f"(num_samples={num_samples}, n_segments={n_segments})"
        )
        return fig, txt

    except Exception as e:
        import traceback
        print(f"LIME audio error: {e}\n{traceback.format_exc()}")
        return None, f"LIME unavailable: {str(e)}"
