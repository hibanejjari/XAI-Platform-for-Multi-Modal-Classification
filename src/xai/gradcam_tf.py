"""
TensorFlow / Keras Grad-CAM for spectrogram CNNs - FIXED
Works with custom CNNs, VGG, MobileNet, ResNet (Conv2D-based)

Fixed:
- Handles models that haven't been called yet
- Better error handling
- Clearer error messages
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model; Grad-CAM requires Conv2D layers.")


def _to_prob_2(model_output: np.ndarray) -> np.ndarray:
    """
    Convert model output to probabilities [P(real), P(fake)].
    Supports:
    - sigmoid: (B,1)
    - softmax: (B,2)
    """
    y = np.array(model_output)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if y.shape[1] == 1:
        # sigmoid: assume it's P(fake)
        p_fake = y
        p_real = 1.0 - p_fake
        return np.concatenate([p_real, p_fake], axis=1)

    if y.shape[1] == 2:
        # already (real,fake) probs/logits
        # if not normalized, normalize
        s = np.sum(y, axis=1, keepdims=True)
        if not np.allclose(s, 1.0, atol=1e-3):
            # apply softmax if looks like logits
            e = np.exp(y - np.max(y, axis=1, keepdims=True))
            y = e / np.sum(e, axis=1, keepdims=True)
        return y

    raise ValueError(f"Unsupported output shape: {y.shape}. Expected (B,1) or (B,2).")


def gradcam_keras(
    model: tf.keras.Model,
    x: np.ndarray,
    target_class: int | None = None,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single sample.

    Args:
        model: tf.keras.Model
        x: input batch, shape (1,H,W,C)
        target_class: 0=Real, 1=Fake. If None, uses predicted class.
        conv_layer_name: optionally specify conv layer name

    Returns:
        cam: (H,W) float in [0,1]
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"x must be shape (1,H,W,C). Got {x.shape}")

    # CRITICAL FIX: Call model once to build it if not built yet
    try:
        # Try to get a layer - this will fail if model not built
        _ = model.layers[0]
    except Exception:
        # Model not built yet, call it once
        print("Building Keras model by calling it once...")
        _ = model(x, training=False)

    if conv_layer_name is None:
        conv_layer_name = _find_last_conv_layer(model)

    try:
        conv_layer = model.get_layer(conv_layer_name)
    except Exception as e:
        raise ValueError(f"Could not find layer '{conv_layer_name}' in model. Error: {e}")

    # Build a model that maps input -> (conv activations, final output)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)

        preds_np = preds.numpy()
        probs = _to_prob_2(preds_np)

        if target_class is None:
            target_class = int(np.argmax(probs[0]))

        # Use logits/probs scalar for the class of interest
        # If preds is (B,1) sigmoid: use that value for Fake, and (1 - val) for Real
        if preds.shape[-1] == 1:
            score = preds[0, 0] if target_class == 1 else (1.0 - preds[0, 0])
        else:
            score = preds[0, target_class]

    grads = tape.gradient(score, conv_out)  # (1,h,w,c)
    if grads is None:
        raise RuntimeError("Gradients are None. Ensure model is differentiable and conv layer is correct.")

    # Global average pooling over spatial dims -> weights per channel
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (1,1,1,c)

    # Weighted sum of conv maps
    cam = tf.reduce_sum(weights * conv_out, axis=-1)  # (1,h,w)
    cam = tf.nn.relu(cam)

    cam_np = cam.numpy()[0]
    # Normalize
    cam_np = cam_np - cam_np.min()
    cam_np = cam_np / (cam_np.max() + 1e-8)

    # Upsample to input size (H,W)
    H, W = x.shape[1], x.shape[2]
    cam_np = tf.image.resize(cam_np[..., None], (H, W), method="bilinear").numpy()[0, :, :, 0]

    cam_np = cam_np - cam_np.min()
    cam_np = cam_np / (cam_np.max() + 1e-8)

    return cam_np