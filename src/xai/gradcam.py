"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
PyTorch 2.x safe implementation using forward hooks and retain_grad
"""

import torch
import torch.nn as nn
import numpy as np


class GradCAM:
    """
    Safer Grad-CAM implementation (no backward hooks).
    Uses forward hook + retain_grad on activations.
    This avoids the "view + inplace" backward hook crash in PyTorch 2.x.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for CAM generation
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = None
        self._register_forward_hook()

    def _register_forward_hook(self):
        """Register forward hook to capture activations"""
        def forward_hook(module, inp, out):
            self.activations = out
            # Ensure grad is retained
            if hasattr(self.activations, "retain_grad"):
                self.activations.retain_grad()
        
        self.hook = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self):
        """Remove registered hooks"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            np.ndarray: Normalized CAM heatmap (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Use predicted class if not specified
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward pass
        self.model.zero_grad(set_to_none=True)
        loss = output[0, target_class]
        loss.backward()

        # Check if activations and gradients were captured
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError(
                "GradCAM did not capture activations/gradients. "
                "Try a different target layer."
            )

        # Get gradients and activations
        grads = self.activations.grad  # [B, C, H, W]
        acts = self.activations        # [B, C, H, W]

        # Global Average Pooling over H,W for weights
        weights = grads.mean(dim=(2, 3), keepdim=True)     # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)    # [B, 1, H, W]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].detach().cpu().numpy()
