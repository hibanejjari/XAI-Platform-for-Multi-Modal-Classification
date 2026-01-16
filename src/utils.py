"""
Utility functions for the Unified XAI Platform
Includes figure to numpy conversion and model fixes
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn




def fig_to_np(fig):
    """
    Convert a matplotlib figure to a numpy RGB image for Gradio Gallery.
    Compatible with Matplotlib 3.10+ (no tostring_rgb).
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        np.ndarray: RGB image array (H, W, 3)
    """
    # Render the canvas
    fig.canvas.draw()

    # Get canvas size
    w, h = fig.canvas.get_width_height()

    # Extract RGBA buffer (new Matplotlib API)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

    # Reshape and drop alpha channel
    img = buf.reshape(h, w, 4)[..., :3]

    # Close figure to free memory
    plt.close(fig)

    return img


def disable_inplace_activations(model: nn.Module):
    """
    Fix PyTorch 2.x Grad-CAM crash:
    'Output ... is a view and is being modified inplace'
    by disabling inplace=True on ReLU/ReLU6/Hardswish.
    
    Args:
        model: PyTorch model to fix
        
    Returns:
        nn.Module: Fixed model
    """
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
            if hasattr(m, "inplace"):
                m.inplace = False
    return model
