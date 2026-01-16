"""
LIME (Local Interpretable Model-agnostic Explanations)
Implementation for image classification
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    lime_image = None
    mark_boundaries = None
    LIME_AVAILABLE = False


def apply_lime_image(model, original_image: Image.Image, device):
    """
    Apply LIME explanation to an image classification model.
    
    Args:
        model: PyTorch classification model
        original_image: PIL Image to explain
        device: torch device (cpu/cuda)
        
    Returns:
        tuple: (matplotlib figure, explanation text) or (None, error message)
    """
    if not LIME_AVAILABLE:
        return None, "LIME not installed. Install: pip install lime scikit-image"

    # Resize image to model input size
    img_array = np.array(original_image.resize((224, 224)))

    def predict_fn(images):
        """Prediction function for LIME"""
        batch = torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])(Image.fromarray(im.astype("uint8")))
            for im in images
        ]).to(device)

        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=200,
    )

    # Get image and mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title("LIME Explanation - Important Regions")
    axes[1].axis("off")

    plt.tight_layout()
    
    return fig, "LIME highlighting decision boundaries"
