"""
SHAP for Keras spectrogram models (KernelExplainer - model agnostic).

Returns SHAP values as an importance heatmap for a single sample.

Note:
- KernelExplainer can be slow. Use small background and small nsamples.
"""

from __future__ import annotations

import numpy as np
import shap


def _to_prob_2(model_output: np.ndarray) -> np.ndarray:
    y = np.array(model_output)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if y.shape[1] == 1:
        p_fake = y
        p_real = 1.0 - p_fake
        return np.concatenate([p_real, p_fake], axis=1)

    if y.shape[1] == 2:
        s = np.sum(y, axis=1, keepdims=True)
        if not np.allclose(s, 1.0, atol=1e-3):
            e = np.exp(y - np.max(y, axis=1, keepdims=True))
            y = e / np.sum(e, axis=1, keepdims=True)
        return y

    raise ValueError(f"Unsupported output shape: {y.shape}. Expected (B,1) or (B,2).")


def shap_audio_kernel(
    model,
    x: np.ndarray,
    target_class: int | None = None,
    background: np.ndarray | None = None,
    nsamples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute SHAP values for a single spectrogram input using KernelExplainer.

    Args:
        model: tf.keras.Model (or any object with .predict)
        x: shape (1,H,W,C)
        target_class: 0=Real, 1=Fake (if None uses predicted class)
        background: optional background batch (B,H,W,C). If None, uses zeros background (10 samples).
        nsamples: shap sampling budget
        seed: random seed

    Returns:
        heatmap: (H,W) normalized [0,1]
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"x must be shape (1,H,W,C). Got {x.shape}")

    rng = np.random.default_rng(seed)
    H, W, C = x.shape[1], x.shape[2], x.shape[3]

    # Flatten inputs for KernelExplainer
    x_flat = x.reshape(1, -1)

    if background is None:
        # Small background of zeros + tiny noise to avoid singularities
        background = np.zeros((10, H, W, C), dtype=np.float32)
        background = background + (rng.normal(0, 1e-3, size=background.shape).astype(np.float32))
    bg_flat = background.reshape(background.shape[0], -1)

    # Choose class
    p0 = _to_prob_2(model.predict(x, verbose=0))
    if target_class is None:
        target_class = int(np.argmax(p0[0]))

    def predict_flat(z_flat: np.ndarray) -> np.ndarray:
        z = z_flat.reshape(z_flat.shape[0], H, W, C).astype(np.float32)
        probs = _to_prob_2(model.predict(z, verbose=0))
        return probs

    explainer = shap.KernelExplainer(predict_flat, bg_flat)

    # shap_values can return list (per class) or array; handle both
    shap_vals = explainer.shap_values(x_flat, nsamples=nsamples)

    if isinstance(shap_vals, list):
        sv = shap_vals[target_class]  # (1, features)
    else:
        # sometimes (1,features,classes) or (1,features)
        sv = shap_vals
        if sv.ndim == 3:
            sv = sv[:, :, target_class]

    sv = np.array(sv).reshape(1, -1)
    sv_img = np.abs(sv[0]).reshape(H, W, C)

    # Reduce channels to heatmap
    heat = sv_img.mean(axis=-1).astype(np.float32)

    # Normalize [0,1]
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    return heat
