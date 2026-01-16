"""
Gradio Interface
Main UI creation and event handling
"""

import gradio as gr
from src.config import (
    AUDIO_MODELS,
    IMAGE_MODELS,
    AUDIO_XAI_METHODS,
    IMAGE_XAI_METHODS,
    XAI_COMPATIBILITY,
)
from src.pipelines import classify_audio, classify_image, compare_xai_for_gallery


def _safe_first(items, fallback=None):
    return items[0] if items else fallback


def create_interface():
    """Create the main Gradio interface"""

    with gr.Blocks(title="Unified XAI Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# XAI Platform for Multi-Modal Classification
**Integrating Audio Deepfake Detection and Chest X-ray Analysis with Explainability**
"""
        )

        # -----------------
        # Helper functions
        # -----------------
        def get_allowed_xai(dtype: str, model_name: str):
            """Return XAI methods allowed for (dtype, model). Falls back safely."""
            allowed = XAI_COMPATIBILITY.get(dtype, {}).get(model_name, [])
            # fallback to global list if mapping missing
            if not allowed:
                allowed = AUDIO_XAI_METHODS if dtype == "Audio" else IMAGE_XAI_METHODS
            return allowed

        def update_inputs(dtype):
            """Update inputs + dropdowns based on data type selection."""
            if dtype == "Audio":
                default_model = _safe_first(AUDIO_MODELS, None)
                allowed_xai = get_allowed_xai("Audio", default_model)
                return (
                    gr.Audio(visible=True),
                    gr.Image(visible=False),
                    gr.Dropdown(choices=AUDIO_MODELS, value=default_model),
                    gr.Dropdown(choices=allowed_xai, value=_safe_first(allowed_xai, None)),
                )
            else:
                default_model = _safe_first(IMAGE_MODELS, None)
                allowed_xai = get_allowed_xai("Image", default_model)
                return (
                    gr.Audio(visible=False),
                    gr.Image(visible=True),
                    gr.Dropdown(choices=IMAGE_MODELS, value=default_model),
                    gr.Dropdown(choices=allowed_xai, value=_safe_first(allowed_xai, None)),
                )

        def update_xai_choices(dtype, model_name):
            """When model changes, re-filter compatible XAI methods."""
            allowed = get_allowed_xai(dtype, model_name)
            return gr.Dropdown(choices=allowed, value=_safe_first(allowed, None))

        def process_classification(dtype, audio, image, model, xai):
            """Run classification pipeline."""
            file = audio if dtype == "Audio" else image
            if file is None:
                return "Please upload a file.", None, ""

            if xai in (None, "", "None"):
                # allow calling pipeline with no XAI
                xai = "None"

            if dtype == "Audio":
                return classify_audio(file, model, xai)
            return classify_image(file, model, xai)

        # -----------------
        # UI
        # -----------------
        with gr.Tabs():
            # -----------------
            # CLASSIFICATION TAB
            # -----------------
            with gr.Tab("📊 Classification"):
                with gr.Row():
                    with gr.Column():
                        data_type = gr.Radio(["Audio", "Image"], label="Data Type", value="Audio")

                        audio_input = gr.Audio(
                            label="Upload Audio File (.wav)",
                            type="filepath",
                            visible=True,
                        )

                        image_input = gr.Image(
                            label="Upload Chest X-Ray Image",
                            type="filepath",
                            visible=False,
                        )

                        model_dropdown = gr.Dropdown(
                            choices=AUDIO_MODELS,
                            label="Classification Model",
                            value=_safe_first(AUDIO_MODELS, None),
                        )

                        # initial XAI choices based on initial Audio + first audio model
                        initial_allowed = XAI_COMPATIBILITY["Audio"].get(
                            model_dropdown.value, AUDIO_XAI_METHODS
                        )
                        xai_dropdown = gr.Dropdown(
                            choices=initial_allowed,
                            label="XAI Method",
                            value=_safe_first(initial_allowed, None),
                        )

                        classify_btn = gr.Button("Classify & Explain", variant="primary")

                    with gr.Column():
                        result_text = gr.Markdown(label="Classification Result")
                        xai_plot = gr.Plot(label="XAI Visualization")
                        xai_explanation = gr.Textbox(label="Explanation", lines=2)

                # when data type changes → swap input + models + allowed XAI
                data_type.change(
                    update_inputs,
                    inputs=[data_type],
                    outputs=[audio_input, image_input, model_dropdown, xai_dropdown],
                )

                # when model changes → re-filter XAI options
                model_dropdown.change(
                    update_xai_choices,
                    inputs=[data_type, model_dropdown],
                    outputs=[xai_dropdown],
                )

                classify_btn.click(
                    process_classification,
                    inputs=[data_type, audio_input, image_input, model_dropdown, xai_dropdown],
                    outputs=[result_text, xai_plot, xai_explanation],
                )

            # -----------------
            # COMPARISON TAB
            # -----------------
            with gr.Tab("🔍 XAI Comparison"):
                gr.Markdown("### Compare Multiple XAI Techniques Side-by-Side")

                with gr.Row():
                    with gr.Column(scale=1):
                        comp_data_type = gr.Radio(["Audio", "Image"], label="Data Type", value="Audio")

                        comp_audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            visible=True,
                        )

                        comp_image_input = gr.Image(
                            label="Upload X-Ray Image",
                            type="filepath",
                            visible=False,
                        )

                        comp_model = gr.Dropdown(
                            choices=AUDIO_MODELS,
                            label="Model",
                            value=_safe_first(AUDIO_MODELS, None),
                        )

                        compare_btn = gr.Button("🔄 Compare XAI Methods", variant="primary")

                    with gr.Column(scale=2):
                        comparison_output = gr.Gallery(
                            label="XAI Comparison Results",
                            columns=2,
                            height="auto",
                        )

                def update_comp_inputs(dtype):
                    if dtype == "Audio":
                        return (
                            gr.Audio(visible=True),
                            gr.Image(visible=False),
                            gr.Dropdown(choices=AUDIO_MODELS, value=_safe_first(AUDIO_MODELS, None)),
                        )
                    else:
                        return (
                            gr.Audio(visible=False),
                            gr.Image(visible=True),
                            gr.Dropdown(choices=IMAGE_MODELS, value=_safe_first(IMAGE_MODELS, None)),
                        )

                comp_data_type.change(
                    update_comp_inputs,
                    inputs=[comp_data_type],
                    outputs=[comp_audio_input, comp_image_input, comp_model],
                )

                def run_comparison(dtype, audio, image, model):
                    file = audio if dtype == "Audio" else image
                    if file is None:
                        return []
                    return compare_xai_for_gallery(file, dtype, model)

                compare_btn.click(
                    run_comparison,
                    inputs=[comp_data_type, comp_audio_input, comp_image_input, comp_model],
                    outputs=[comparison_output],
                )

            # -----------------
            # ABOUT TAB
            # -----------------
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
## Project Overview

This unified platform integrates two Explainable AI systems:

### 🎵 Audio: Deepfake Detection
- **Models:** Custom CNN, MobileNet, VGG16
- **Dataset:** Fake-or-Real (FoR)
- **XAI Methods:** Grad-CAM, SHAP, LIME

### 🫁 Image: Chest X-ray Analysis (CheXpert)
- **Models:** DenseNet121 (CheXpert-pretrained via TorchXRayVision) + optional AlexNet/DenseNet baselines
- **Dataset:** CheXpert-style chest X-rays
- **XAI Methods:** Grad-CAM, LIME

### ✨ Features
- Multi-modal input support (Audio & Image)
- Multiple classification models
- Automatic XAI filtering based on data type + model compatibility
- Side-by-side XAI comparison
- Visual explanations for predictions

### 📋 How to Use
1. **Classification Tab:** Upload audio or image, select model and XAI method
2. **Comparison Tab:** Compare multiple XAI techniques on the same input
3. View results, heatmaps, and explanations

### Team Information 
- Lisa NACCACHE
- Hiba NEJJARI 
- Neil MAHCER
- Wendy DUONG 
- Cyprien MOUTON 
"""
                )

    return demo
