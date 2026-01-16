"""
XAI comparison pipeline - CORRECTED VERSION
Fixed to properly display side-by-side comparisons in Gradio Gallery
Based on working Streamlit implementations from reference repos
"""

from src.config import AUDIO_XAI_METHODS, IMAGE_XAI_METHODS
from src.pipelines.classify_audio import classify_audio
from src.pipelines.classify_image import classify_image
from src.utils import fig_to_np
import matplotlib.pyplot as plt
import numpy as np


def compare_xai_methods(file: str, data_type: str, model_name: str):
    """
    Compare multiple XAI methods on the same input.
    FIXED: Ensures all figures are properly generated and converted.
    
    Args:
        file: Path to input file (audio or image)
        data_type: Type of data ("Audio" or "Image")
        model_name: Name of the classification model
        
    Returns:
        list: List of (method_name, figure, explanation_text) tuples
    """
    results = []
    
    if data_type == "Audio":
        # Compare audio XAI methods
        for method in AUDIO_XAI_METHODS:
            try:
                _, fig, txt = classify_audio(file, model_name, method)
                if fig is not None:
                    results.append((method, fig, txt))
                else:
                    # Create error placeholder figure
                    fig_err = plt.figure(figsize=(10, 4))
                    plt.text(0.5, 0.5, f"{method} failed to generate visualization",
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    results.append((method, fig_err, f"{method}: Error"))
            except Exception as e:
                print(f"Error with {method}: {e}")
                # Create error placeholder
                fig_err = plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, f"{method} error: {str(e)}",
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                results.append((method, fig_err, f"{method}: Error"))
    else:
        # Compare image XAI methods
        for method in IMAGE_XAI_METHODS:
            try:
                _, fig, txt = classify_image(file, model_name, method)
                if fig is not None:
                    results.append((method, fig, txt))
                else:
                    # Create error placeholder figure
                    fig_err = plt.figure(figsize=(10, 4))
                    plt.text(0.5, 0.5, f"{method} failed to generate visualization",
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    results.append((method, fig_err, f"{method}: Error"))
            except Exception as e:
                print(f"Error with {method}: {e}")
                # Create error placeholder
                fig_err = plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, f"{method} error: {str(e)}",
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                results.append((method, fig_err, f"{method}: Error"))
    
    return results


def compare_xai_for_gallery(file: str, data_type: str, model_name: str):
    """
    Compare XAI methods and format for Gradio Gallery.
    FIXED: Properly converts figures to numpy arrays and handles captions.
    
    Args:
        file: Path to input file
        data_type: Type of data ("Audio" or "Image")
        model_name: Name of the model
        
    Returns:
        list: List of (image_array, caption) tuples for Gallery
    """
    if file is None:
        return []
    
    results = compare_xai_methods(file, data_type, model_name)
    
    gallery_items = []
    for method, fig, txt in results:
        if fig is None:
            continue
        
        try:
            # Convert matplotlib figure to numpy array
            img_array = fig_to_np(fig)
            
            # Create caption with method name and short explanation
            caption = f"{method}"
            if txt:
                # Truncate long explanations for caption
                short_txt = txt[:100] + "..." if len(txt) > 100 else txt
                caption = f"{method}: {short_txt}"
            
            gallery_items.append((img_array, caption))
        except Exception as e:
            print(f"Error converting figure for {method}: {e}")
            # Create error image
            error_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
            gallery_items.append((error_img, f"{method}: Conversion Error"))
    
    return gallery_items


def compare_xai_with_details(file: str, data_type: str, model_name: str):
    """
    Compare XAI methods and return detailed results including prediction.
    ENHANCED: Returns prediction result along with comparisons.
    
    Args:
        file: Path to input file
        data_type: Type of data ("Audio" or "Image")
        model_name: Name of the model
        
    Returns:
        tuple: (prediction_text, gallery_items)
    """
    if file is None:
        return "Please upload a file", []
    
    # Get prediction first (using the first available XAI method)
    prediction_text = "Classification Results:\n"
    
    try:
        if data_type == "Audio":
            first_method = AUDIO_XAI_METHODS[0]
            pred_result, _, _ = classify_audio(file, model_name, first_method)
            prediction_text += pred_result
        else:
            first_method = IMAGE_XAI_METHODS[0]
            pred_result, _, _ = classify_image(file, model_name, first_method)
            prediction_text += pred_result
    except Exception as e:
        prediction_text += f"Error getting prediction: {e}"
    
    # Get all XAI comparisons
    gallery_items = compare_xai_for_gallery(file, data_type, model_name)
    
    return prediction_text, gallery_items


def create_side_by_side_comparison(file: str, data_type: str, model_name: str):
    """
    Create a single image with side-by-side comparison of all XAI methods.
    ALTERNATIVE: For users who prefer a single combined image.
    
    Args:
        file: Path to input file
        data_type: Type of data ("Audio" or "Image")
        model_name: Name of the model
        
    Returns:
        matplotlib figure with all methods side by side
    """
    if file is None:
        return None
    
    results = compare_xai_methods(file, data_type, model_name)
    
    if not results:
        return None
    
    # Determine number of methods
    n_methods = len(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    # Handle single method case
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method, method_fig, txt) in enumerate(results):
        if method_fig is not None:
            # Convert the method figure to array
            img_array = fig_to_np(method_fig)
            axes[idx].imshow(img_array)
            axes[idx].set_title(method, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"{method}\nNot Available",
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    return fig