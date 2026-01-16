"""
LIME-style explanation for Keras spectrogram models.

This is a minimal, dependency-light LIME variant for spectrogram images:
- Segment spectrogram into a grid of patches
- Sample random binary masks to hide patches
- Fit weighted linear regression to approximate local decision boundary
- Return patch-importance map as heatmap

Works for any Keras model with predict().
"""

from __future__ import annotations

import numpy as np


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


def lime_audio_spectrogram(
    model,
    x: np.ndarray,
    target_class: int | None = None,
    n_samples: int = 600,
    grid: tuple[int, int] = (8, 8),
    hide_value: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Produce an importance heatmap over the spectrogram.

    Args:
        model: tf.keras.Model (or any object with .predict)
        x: input batch shape (1,H,W,C)
        target_class: 0=Real,1=Fake. If None uses predicted class.
        n_samples: number of perturbations
        grid: (gh, gw) patches
        hide_value: value to replace hidden patches
        seed: random seed

    Returns:
        heatmap: shape (H,W) normalized [0,1]
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"x must be shape (1,H,W,C). Got {x.shape}")

    rng = np.random.default_rng(seed)

    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    gh, gw = grid

    # Determine patch boundaries
    ys = np.linspace(0, H, gh + 1, dtype=int)
    xs = np.linspace(0, W, gw + 1, dtype=int)

    n_features = gh * gw

    # Original prediction
    p0 = _to_prob_2(model.predict(x, verbose=0))
    if target_class is None:
        target_class = int(np.argmax(p0[0]))

    # Sample binary masks (1 = keep, 0 = hide)
    masks = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.int32)

    # Ensure the first sample is "all on" (no masking)
    masks[0, :] = 1

    # Generate perturbed inputs
    Xp = np.repeat(x, repeats=n_samples, axis=0)  # (n,H,W,C)
    for i in range(n_samples):
        m = masks[i].reshape(gh, gw)
        for r in range(gh):
            for c in range(gw):
                if m[r, c] == 0:
                    y0, y1 = ys[r], ys[r + 1]
                    x0, x1 = xs[c], xs[c + 1]
                    Xp[i, y0:y1, x0:x1, :] = hide_value

    # Predict
    probs = _to_prob_2(model.predict(Xp, verbose=0))  # (n,2)
    y = probs[:, target_class]  # (n,)

    # LIME kernel weights based on Hamming distance
    d = np.mean(masks == 0, axis=1)  # fraction hidden
    sigma = 0.25
    w = np.exp(-(d ** 2) / (sigma ** 2))  # (n,)

    # Weighted linear regression (closed-form ridge)
    # Add bias term
    A = np.concatenate([np.ones((n_samples, 1)), masks.astype(np.float32)], axis=1)  # (n, 1+f)
    Wdiag = w.astype(np.float32)

    # Ridge for stability
    lam = 1e-3
    Aw = A * Wdiag[:, None]
    yw = y * Wdiag

    # Solve (A^T W A + lam I) beta = A^T W y
    ATA = Aw.T @ A
    ATy = Aw.T @ yw
    ATA = ATA + lam * np.eye(ATA.shape[0], dtype=np.float32)

    beta = np.linalg.solve(ATA, ATy)  # (1+f,)

    # Feature weights = beta[1:]
    feat_w = beta[1:].reshape(gh, gw)

    # Convert patch weights to pixel heatmap
    heat = np.zeros((H, W), dtype=np.float32)
    for r in range(gh):
        for c in range(gw):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            heat[y0:y1, x0:x1] = feat_w[r, c]

    # Normalize to [0,1]
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    return heat
