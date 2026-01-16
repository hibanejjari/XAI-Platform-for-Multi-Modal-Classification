"""
UI Components
Reusable Gradio component builders and logic
"""

import gradio as gr
from src.config import (
    AUDIO_MODELS, IMAGE_MODELS,
    AUDIO_XAI_METHODS, IMAGE_XAI_METHODS
)


def create_data_type_selector():
    """Create data type radio selector"""
    return gr.Radio(
        ["Audio", "Image"],
        label="Data Type",
        value="Audio"
    )


def create_model_dropdown(initial_type="Audio"):
    """Create model selection dropdown"""
    choices = AUDIO_MODELS if initial_type == "Audio" else IMAGE_MODELS
    return gr.Dropdown(
        choices=choices,
        label="Classification Model",
        value=choices[0]
    )


def create_xai_dropdown(initial_type="Audio"):
    """Create XAI method selection dropdown"""
    choices = AUDIO_XAI_METHODS if initial_type == "Audio" else IMAGE_XAI_METHODS
    return gr.Dropdown(
        choices=choices,
        label="XAI Method",
        value=choices[0]
    )


def update_inputs_for_data_type(dtype: str):
    """
    Update UI inputs based on selected data type.
    
    Args:
        dtype: Data type ("Audio" or "Image")
        
    Returns:
        dict: Dictionary of component updates
    """
    if dtype == "Audio":
        return {
            "audio_visible": True,
            "image_visible": False,
            "model_choices": AUDIO_MODELS,
            "model_value": AUDIO_MODELS[0],
            "xai_choices": AUDIO_XAI_METHODS,
            "xai_value": AUDIO_XAI_METHODS[0],
        }
    else:
        return {
            "audio_visible": False,
            "image_visible": True,
            "model_choices": IMAGE_MODELS,
            "model_value": IMAGE_MODELS[0],
            "xai_choices": IMAGE_XAI_METHODS,
            "xai_value": IMAGE_XAI_METHODS[0],
        }
