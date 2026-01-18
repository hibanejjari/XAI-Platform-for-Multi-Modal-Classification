# Unified XAI Platform for Multi-Modal Classification

**Integrating Audio Deepfake Detection and Chest X-ray Analysis with Explainability**
---

## Team ( ESILV A5 : DIA4 )

**Lisa NACCACHE** • **Hiba NEJJARI** • **Neil MAHCER** • **Wendy DUONG** • **Cyprien MOUTON**

---

## Table of Contents


- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Quick Start](#quick-start)
- [Demo](#demo)
- [Technical Report](#technical-report)
  - [Design and Integration Decisions](#design-and-integration-decisions)
    - [Framework Migration: Streamlit → Gradio](#framework-migration-streamlit--gradio)
    - [Dual Framework Support (PyTorch + TensorFlow)](#dual-framework-support-pytorch--tensorflow)
    - [Automatic XAI Compatibility System](#automatic-xai-compatibility-system)
    - [Model Management Strategy](#model-management-strategy)
    - [Standardized Visualization](#standardized-visualization)
  - [Selected Models and XAI Methods](#selected-models-and-xai-methods)
    - [Audio Models](#audio-models)
    - [Image Models](#image-models)
    - [XAI Implementation Details](#xai-implementation-details)
  - [Improvements Over Original Repositories](#improvements-over-original-repositories)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Generative AI Usage](#generative-ai-usage)
- [References](#references)

---

## Overview
We developed a unified interactive platform that integrates two explainable AI (XAI) systems, audio deepfake detection and lung cancer detection from medical images, into a single interface. The application is organized into two main tabs:
- a **Classification page** where users select the input type (audio or image), choose a pretrained model, and apply an appropriate XAI technique to visualize the explanation alongside the prediction
- an **XAI Comparison page** that enables side-by-side visualization of different XAI methods for the model chosen based on the input -> it's an interface layer that automatically manages compatibility by filtering out XAI techniques that are not applicable to the selected data modality and that allows to see details of the selected method.

The platform supports multiple models and required XAI methods (Grad-CAM, LIME, SHAP) respectively to the input type and model chosen ( VGG16, MobileNet, Custom CNN, FoR Keras (TensorFlow) for audios and XRV DenseNet121 (CheXpert), AlexNet, DenseNet for images) .

The goal of this project is to gain practical experience with Explainable AI (XAI) for audio and image data, while introducing a multi-modal framework designed for future extensibilit. whether its through the models used or to support additional input types .
### Technologies Used

- **Deep Learning Models**: VGG16, MobileNetV2, Custom CNN, DenseNet121, AlexNet  
  *(additional models explored: XRV DenseNet121 (TorchXRayVision), FoR Keras model)*

- **Explainable AI (XAI) Techniques**  : Grad-CAM, LIME, SHAP

- **Programming & Libraries**: Python, PyTorch, TensorFlow/Keras, Pillow, OpenCV (cv2), NumPy, Matplotlib
  
- **Datasets**: Fake-or-Real (Kaggle, audio deepfake), CheXpert (Stanford ML Group chest X-rays)

- **Web Application Framework**: Gradio 4.0+

## Quick Start

For detailed installation instructions, configuration options, as well as usage guide and usage examples, please see **[Setup_Usage.md](SETUP.md)**.

**Quick Overview:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Interface launches at `http://127.0.0.1:7860`

---

## Demo

To view the demonstration video, open and download the mp4 file.
[Watch the demo video](demo.mp4)

## Technical Report

### Design and Integration Decisions

Our goal was to integrate two independent XAI systems into a single platform with shared infrastructure for model loading, preprocessing, and visualization while maintaining modality-specific pipelines. Audio and image workflows operate independently but share common XAI components, enabling code reuse without sacrificing domain optimization. The decisions on design and integration are as follows :

#### Framework Migration: Streamlit → Gradio

The original repository for audio deepfakes used Streamlit, but we migrated to Gradio for better Python compatibility. Although Streamlit provides similar functionality, Gradio was selected because its API is more directly oriented toward model-centric workflows, which simplified the implementation of interactive inference and XAI visualizations in a multi-modal setting.

#### Dual Framework Support (PyTorch + TensorFlow)

Supporting both frameworks required implementing duplicate XAI methods: PyTorch versions use hooks and gradient extraction, TensorFlow versions use `tf.GradientTape`. The centralized `ModelManager` automatically detects framework type and routes to appropriate implementations, maintaining a unified interface.

#### Automatic XAI Compatibility System

Configuration mappings define valid XAI-model combinations:
- Audio models (Custom CNN, VGG16, MobileNet): Grad-CAM, SHAP, LIME
- XRV DenseNet121: Grad-CAM only (LIME fails due to channel mismatch)
- Baseline image models (AlexNet, DenseNet): Grad-CAM, LIME

The UI dynamically filters dropdown options based on these mappings.

#### Model Management Strategy

The 'ModelManager' centralizes model handling by loading models only when needed, reusing already loaded models to avoid redundant initialization, automatically selecting the available computation device (CPU or GPU), and providing a common interface for both PyTorch and TensorFlow models.

#### Standardized Visualization

All XAI methods produce consistent 3-panel visualizations (original input, heatmap, overlay) with clearly labeled method names, regardless of framework or technique enabling meaningful side-by-side comparison.

---

### Selected Models and XAI Methods

#### Audio Models

**Custom CNN (PyTorch)** : 3 convolutional layers (32→64→128 filters), processes (1, 128, 128) mel-spectrograms for binary real/fake classification. Fully functional with Grad-CAM, SHAP, LIME support.

**VGG16 & MobileNetV2 (PyTorch)** : Adapted from torchvision, trained without ImageNet pre-training : VGG16 provides depth, MobileNetV2 offers speed. Both fully operational with all XAI methods.

**FoR Keras (TensorFlow)** : Custom CNN for variable-length mel-spectrograms. Architecture complete (11 MB `.keras` file), but training incomplete due to resource constraints (note : FoR dataset requires 5GB space, if 16GB+ RAM; to have available: 8GB RAM). Despite untrained state, it shows our framework's capability with functional TensorFlow XAI implementations.

#### Image Models

**XRV DenseNet121 (PyTorch - TorchXRayVision)** : Pre-trained on CheXpert for 14 pathology predictions. Accepts grayscale (1, 224, 224) X-rays. Weights exported (27.8 MB `.pth`), but full training incomplete (note : CheXpert: 439GB, requires 32GB+ RAM). Grad-CAM works; LIME incompatible due to channel mismatch (expects 1-channel grayscale, receives 3-channel RGB). Outputs generic labels ("Pathology_16") instead of medical terms ("Cardiomegaly") due to incomplete training.

**AlexNet & DenseNet121 (PyTorch)** : ImageNet pre-trained, modified for binary classification (Normal/Malignant). Fully functional with Grad-CAM and LIME, demonstrating LIME channel issue is XRV-specific.

#### XAI Implementation Details

**Grad-CAM** : Hook-based (PyTorch) or `tf.GradientTape` (TensorFlow) gradient extraction. Automatic target layer detection. Works on all convolutional models. 

**LIME** : 8×8 grid segmentation for spectrograms, superpixel segmentation (quickshift) for images. 200-600 perturbation samples. Model-agnostic but requires channel compatibility. Works on audio models and baseline image models; fails on XRV DenseNet121. 

**SHAP** : KernelExplainer for both frameworks with 100-200 coalition samples. Currently audio-only (image implementation in development). 

---

### Improvements Over Original Repositories

**Original: [Deepfake Audio Detection](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI) & [Lung Cancer Detection](https://github.com/schaudhuri16/LungCancerDetection)**

Building on the original implementations, this project retains their initial design choices by starting out on the same training.

We extend them by re-implementing the pipelines in PyTorch.
 
We especially integrate both the audio and image tasks into a single dynamic unified platform where we can compare the XAI method based on the model interactively chosen. 

To summarize : 

**Our Platform:**
- Unified interface for both modalities
- Dual framework support (PyTorch + TensorFlow)
- Complete XAI coverage
- Automatic compatibility filtering prevents invalid combinations
- Side-by-side XAI comparison mode
- Centralized model manager with lazy loading and caching
- Enhanced preprocessing: variable-length audio support, automatic channel adaptation (grayscale ↔ RGB)
- Modern Gradio UI with better Python compatibility
- Professional 3-panel visualizations with consistent formatting
- Comprehensive error handling with user-friendly messages
- Modular architecture for easy extension


---

## Known Limitations

**XRV DenseNet121 LIME Channel Mismatch**
- Error: "expected input to have 1 channels, but got 3 channels instead"
- Cause: XRV expects grayscale (1-channel), LIME passes RGB (3-channel)
- Scope: XRV-specific issue; LIME works correctly on AlexNet and DenseNet121

**XRV Generic Labels**
- Displays "Pathology_16" instead of "Cardiomegaly"
- Cause: Training incomplete, label file mismatch (14 outputs vs 18 labels)
- Workaround: Grad-CAM still provides accurate anatomical localization

**FoR Keras Untrained**
- Model architecture complete but produces random predictions
- Cause: No training performed (caused by hardware issues since the dataset size is 5GB, RAM required: 16GB+ and inr our case the RAM available is 8GB)
- Status: Training script prepared, infrastructure operational 

**XAI Method and Model-Specific Limitations**
- **Audio Models (PyTorch XAI, e.g., SHAP, LIME)**: Explanations are generated using region-based masking on mel-spectrograms (SLIC segmentation with a limited number of regions and on/off perturbations). This produces coarse, blob-like attributions that average importance over large time–frequency areas and rely on zero-masked backgrounds, which can create out-of-distribution inputs. Consequently, highlighted regions may not precisely match perceptually relevant audio events and should be interpreted qualitatively rather than as fine-grained temporal localization.
- **Image Models (PyTorch XAI, e.g., LIME)**: Explanations are based on superpixel segmentation driven by pixel similarity rather than anatomical structure. As a result, highlighted regions may not align with clinically meaningful anatomy, and masking can disrupt spatial context important for medical interpretation. Even when execution succeeds, these explanations provide coarse intuition and are less reliable than gradient-based methods such as Grad-CAM for medical imaging tasks.


---
## Future Work
- Complete training of FoR Keras and XRV DenseNet121 with sufficient computational resources
- Resolve LIME compatibility for grayscale medical images
- Improve temporal precision of audio XAI by refining region segmentation
- Extend SHAP support to image models
- Explore anatomy-aware and gradient-based XAI alternatives
---
## Generative AI Usage

We used **Claude 3.5 Sonnet** (Anthropic) and **ChatGPT** (OpenAI) for debugging framework integration, implementing XAI methods (gradient computation, segmentation algorithms), error resolution (channel mismatches, model loading), code refactoring, documentation writing, and architecture review.

The team independently handled project design (architecture, workflow, model selection), core implementation (model integration, preprocessing, UI, configuration), and validation (testing, XAI verification, benchmarking). All AI-generated code was reviewed, tested, and modified to meet requirements. We understand the algorithms and take full responsibility for implementation.

---

## References

**Source Repositories:**
1. Deepfake Audio Detection with XAI - https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI
2. Lung Cancer Detection - https://github.com/schaudhuri16/LungCancerDetection
3. TorchXRayVision - https://github.com/mlmed/torchxrayvision

**Datasets:**
4. Fake-or-Real (FoR) - https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
5. CheXpert - https://www.kaggle.com/datasets/ashery/chexpert | https://stanfordmlgroup.github.io/competitions/chexpert/

**XAI Papers:**
6. Grad-CAM - Selvaraju et al., ICCV 2017 - https://arxiv.org/abs/1610.02391
7. LIME - Ribeiro et al., KDD 2016 - https://arxiv.org/abs/1602.04938
8. SHAP - Lundberg & Lee, NeurIPS 2017 - https://arxiv.org/abs/1705.07874

**Libraries:**
9. Gradio - https://gradio.app/ | PyTorch - https://pytorch.org/ | TensorFlow - https://tensorflow.org/ | Librosa - https://librosa.org/

---


<div align="center">

**Made with care by the DIA4 Team**  
*Explainable AI for Audio & Medical Imaging*

[![Python](https://img.shields.io/badge/Python-3.8--3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

</div>
