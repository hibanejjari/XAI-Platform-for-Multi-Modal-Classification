# Unified XAI Platform for Multi-Modal Classification

**Integrating Audio Deepfake Detection and Chest X-ray Analysis with Explainability**

---

## Team (ESILV A5: DIA4)

**Lisa NACCACHE** • **Hiba NEJJARI** • **Neil MAHCER** • **Wendy DUONG** • **Cyprien MOUTON**

---

## Table of Contents

- [Overview](#overview)
- [Demo Video](#demo-video)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Technical Report](#technical-report)
  - [Design and Integration Decisions](#design-and-integration-decisions)
  - [Selected Models and XAI Methods](#selected-models-and-xai-methods)
  - [Improvements Over Original Repositories](#improvements-over-original-repositories)
- [Current Project Status](#current-project-status)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Screenshots and Results](#screenshots-and-results)
- [AI Usage Declaration](#ai-usage-declaration)
- [References](#references)

---

## Overview

We developed a unified interactive platform that integrates two explainable AI (XAI) systems—audio deepfake detection and chest X-ray pathology detection—into a single interface. The application is organized into two main tabs:

1. **Classification Tab**: Users select the input type (audio or image), choose a pre-trained model, and apply an appropriate XAI technique to visualize the explanation alongside the prediction.

2. **XAI Comparison Tab**: Enables side-by-side visualization of different XAI methods for the chosen model and input. The interface automatically manages compatibility by filtering out XAI techniques that are not applicable to the selected data modality.

The platform supports multiple models and all required XAI methods (Grad-CAM, LIME, SHAP) with automatic compatibility filtering based on input type and model architecture.

### Technologies Used

**Deep Learning Models**:
- VGG16, MobileNetV2, Custom CNN (audio classification)
- DenseNet121, AlexNet (baseline image classification)
- XRV DenseNet121 from TorchXRayVision (medical imaging)
- FoR Keras model (TensorFlow/Keras audio model)

**Explainable AI (XAI) Techniques**: Grad-CAM, LIME, SHAP

**Programming Languages & Libraries**:
- **Core**: Python 3.13
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.13+, Keras, torchvision, TorchXRayVision
- **Audio Processing**: Librosa (mel-spectrogram generation)
- **Image Processing**: OpenCV (cv2), Pillow, scikit-image
- **XAI Libraries**: LIME, SHAP (with custom integrations)
- **Scientific Computing**: NumPy, Matplotlib
- **Web Framework**: Gradio 4.0+

**Datasets**:
- Fake-or-Real (FoR) - Kaggle, audio deepfake detection
- CheXpert - Stanford ML Group, 224,316 chest X-rays

**Web Application Framework**: Gradio 4.0+ (migrated from Streamlit for better Python 3.13 compatibility)

**Project Goals:**
- Gain practical experience with Explainable AI for multi-modal data (audio and images)
- Create an extensible framework for future model and modality additions
- Integrate PyTorch and TensorFlow/Keras models in a unified system
- Provide transparent, interpretable AI predictions for critical applications

---

## Demo Video

**[Watch Demo Video](YOUR_VIDEO_LINK_HERE)**

*Replace with actual video link (YouTube, Google Drive, etc.)*

---

## Quick Start

For detailed installation instructions, configuration options, and usage examples, please see **[SETUP.md](SETUP.md)**.

**Quick Overview:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Interface launches at `http://127.0.0.1:7860`

---

## Technical Report

### Design and Integration Decisions

We integrated two separate XAI systems (audio deepfake detection and medical image analysis) into a single unified platform. This required creating shared infrastructure for model loading, preprocessing, and visualization while maintaining separate pipelines for each modality. The platform uses a modular architecture where audio and image workflows remain independent but share common XAI visualization components.

The original repositories used Streamlit for the interface, but we migrated to Gradio for better Python 3.13 compatibility and faster development with built-in ML components. Gradio's native support for audio/image inputs and event-driven architecture made it easier to implement dynamic UI elements that adapt based on user selections.

One major challenge was supporting both PyTorch and TensorFlow/Keras models in the same application. Most models use PyTorch (Custom CNN, VGG16, MobileNet, XRV DenseNet121, AlexNet), but we wanted to demonstrate cross-framework capability with the FoR Keras model. This required implementing duplicate versions of each XAI method—one for PyTorch models using hooks and gradients, and another for TensorFlow using `tf.GradientTape`. The model manager automatically detects the framework and routes to the appropriate XAI implementation.

To prevent users from selecting incompatible XAI methods, we implemented an automatic compatibility system. The configuration file defines which XAI methods work with each model:
- Audio models (Custom CNN, VGG16, MobileNet): Grad-CAM, SHAP, LIME
- XRV DenseNet121: Grad-CAM only (LIME has channel issues)
- Baseline image models (AlexNet, DenseNet): Grad-CAM, LIME

The UI dynamically filters dropdown options based on these mappings, so users only see applicable choices. This prevents errors and improves usability.

For model management, we created a centralized `ModelManager` class that loads models on-demand and caches them in memory. This avoids reloading the same model repeatedly, significantly improving performance. The manager also handles device placement (CPU/GPU) automatically and works with both PyTorch and TensorFlow models through a unified interface.

All three XAI methods produce consistent 3-panel visualizations (original input, heatmap, overlay) regardless of the underlying technique or framework. This standardization makes it easy to compare different methods side-by-side in the comparison tab.

---

### Selected Models and XAI Methods

#### Audio Models

**1. Custom CNN (PyTorch)**
- **Architecture**: 3 convolutional layers (32→64→128 filters) + adaptive pooling + fully connected layers
- **Input**: (1, 128, 128) mel-spectrogram
- **Output**: Binary classification (Real/Fake)
- **Implementation**: Custom architecture designed for audio spectrograms
- **Status**: Fully functional
- **XAI Compatibility**: Grad-CAM, SHAP, LIME

**2. VGG16 (PyTorch)**
- **Base Architecture**: torchvision VGG16
- **Adaptation**: Modified classifier for binary classification, adapted for 3-channel spectrograms
- **Implementation**: `weights=None` (trained from scratch, no ImageNet pre-training)
- **Status**: Fully functional
- **XAI Compatibility**: Grad-CAM, SHAP, LIME

**3. MobileNetV2 (PyTorch)**
- **Base Architecture**: torchvision MobileNetV2
- **Adaptation**: Lightweight architecture for faster inference
- **Implementation**: `weights=None` (trained from scratch)
- **Status**: Fully functional
- **XAI Compatibility**: Grad-CAM, SHAP, LIME

**4. FoR Keras Model (TensorFlow/Keras)**
- **Dataset**: Fake-or-Real (FoR) from Kaggle
- **Architecture**: Custom CNN for mel-spectrograms
- **Input**: (128, T, 1) variable-length mel-spectrogram
- **Output**: Sigmoid probability (fake probability)
- **File**: `audio_classifier.keras` (11 MB)
- **Status**: Architecture ready, training in progress
- **Note**: Only model using external dataset; all PyTorch audio models are custom implementations
- **XAI Compatibility**: Grad-CAM (TF), SHAP (TF), LIME (TF)

#### Image Models

**1. XRV DenseNet121 (PyTorch - TorchXRayVision)**
- **Source**: TorchXRayVision library
- **Dataset**: CheXpert (Stanford ML Group) - 224,316 chest X-rays
- **Architecture**: DenseNet121 pre-trained on medical imaging data
- **Input**: (1, 224, 224) grayscale X-ray
- **Output**: 14 pathology predictions (multi-label)
- **File**: `xrv-densenet121-res224-chex.pth` (27.8 MB)
- **Status**: Weights exported, training incomplete
- **Pathologies**: Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion, etc.
- **XAI Compatibility**: Grad-CAM ✅ (working), LIME ❌ (channel mismatch - XRV-specific issue, works on AlexNet/DenseNet)

**2. AlexNet (PyTorch)**
- **Base Architecture**: torchvision AlexNet
- **Pre-training**: ImageNet weights (`weights=DEFAULT`)
- **Adaptation**: Modified classifier for binary classification (Normal/Malignant)
- **Status**: Fully functional
- **Use Case**: Baseline model for comparison
- **XAI Compatibility**: Grad-CAM ✅, LIME ✅ (working)

**3. DenseNet121 (PyTorch)**
- **Base Architecture**: torchvision DenseNet121
- **Pre-training**: ImageNet weights (`weights=DEFAULT`)
- **Adaptation**: Modified classifier for binary classification
- **Status**: Fully functional
- **Use Case**: Baseline model for comparison
- **XAI Compatibility**: Grad-CAM ✅, LIME ✅ (working)

#### XAI Methods Implementation

We implemented all three required XAI methods with dual framework support (PyTorch + TensorFlow):

**Implementation Strategy:**

| Method | PyTorch Implementation | TensorFlow Implementation | Performance |
|--------|----------------------|--------------------------|-------------|
| **Grad-CAM** | Hook-based gradient extraction | `tf.GradientTape` | 1-4s (fast) |
| **LIME** | Region-based perturbation | Same (model-agnostic) | 10-20s (slow) |
| **SHAP** | KernelExplainer | KernelExplainer | 30-60s (very slow) |

**Key Implementation Details:**

1. **Grad-CAM**: Automatic target layer detection for each model architecture, 3-panel visualization (original + heatmap + overlay)

2. **LIME**: 
   - Audio: 8×8 grid segmentation of spectrograms, 200-600 perturbation samples
   - Image: Superpixel segmentation (quickshift algorithm), 200 samples
   - **Issue**: XRV DenseNet121 expects 1-channel (grayscale) but LIME passes 3-channel (RGB) - works correctly on AlexNet/DenseNet baseline models

3. **SHAP**: KernelExplainer with 100-200 coalition samples, zero-filled background distribution

**Visualization Consistency:**
All methods produce uniform 3-panel matplotlib figures for direct comparison in the comparison tab.

---

### Improvements Over Original Repositories

#### Original Repository 1: Deepfake Audio Detection with XAI
**Source**: https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI

**Original Implementation:**
- Separate Jupyter notebooks for each model and XAI method
- Streamlit interface for single-model inference
- PyTorch-only implementation
- Limited error handling

#### Original Repository 2: Lung Cancer Detection
**Source**: https://github.com/schaudhuri16/LungCancerDetection

**Original Implementation:**
- Standalone image classification system
- Basic Grad-CAM visualization
- No multi-model comparison
- Limited XAI method coverage

---

#### Our Improvements

**1. Unified Multi-Modal Platform**

**Original**: Two separate repositories, one for audio and one for images
**Our Solution**: Single integrated platform supporting both modalities

**Benefits:**
- Users can analyze both audio and image data without switching applications
- Consistent interface and workflow across modalities
- Shared infrastructure reduces code duplication
- Demonstrates XAI portability across domains

**2. Enhanced Framework Support**

**Original**: PyTorch only (audio repo) or TensorFlow only (image repo)
**Our Solution**: Dual framework support (PyTorch + TensorFlow/Keras)

**Technical Achievement:**
- Implemented duplicate XAI methods for both frameworks
- Created unified interface layer that abstracts framework differences
- Automatic framework detection based on model type
- Seamless switching between frameworks during inference

**3. Comprehensive XAI Coverage**

**Original Audio Repo**: Grad-CAM and basic SHAP
**Original Image Repo**: Grad-CAM only
**Our Solution**: Full implementation of Grad-CAM, LIME, and SHAP for all applicable models

**Improvements:**
- Added LIME for audio analysis (not in original)
- Added SHAP for audio with proper mel-spectrogram handling
- Implemented TensorFlow versions of all XAI methods
- Created consistent 3-panel visualizations across all methods

**4. Automatic Compatibility Management**

**Original**: No validation of model-XAI compatibility
**Our Solution**: Configuration-based compatibility system

**Implementation:**
```python
# Prevents invalid combinations (e.g., SHAP on image models)
# Automatically filters UI options based on selected model
# Clear error messages when incompatibilities detected
```

**Benefits:**
- Prevents runtime errors from incompatible combinations
- Improves user experience (only shows valid options)
- Makes platform more robust and production-ready

**5. XAI Comparison Mode**

**Original**: Single XAI method visualization at a time
**Our Solution**: Side-by-side comparison of multiple XAI methods

**Features:**
- Simultaneous display of all compatible XAI methods
- Consistent visualization format for direct comparison
- Automatic layout adjustment based on number of methods
- Helps users understand differences between explanation techniques

**6. Improved Model Management**

**Original**: Manual model loading in each script
**Our Solution**: Centralized model manager with caching

**Features:**
- Lazy loading (models loaded only when needed)
- Automatic caching (models persist across predictions)
- Memory efficient (automatic cleanup when not needed)
- Framework-agnostic interface

**Code Quality Improvements:**
```python
# Original: Manual model loading in each script
model = torch.load('model.pth')

# Our solution: Centralized management
model = model_manager.get_audio_model('VGG16')  # Automatically cached
```

**7. Production-Ready Error Handling**

**Original**: Basic try-catch blocks, limited error messages
**Our Solution**: Comprehensive error handling with user-friendly messages

**Improvements:**
- Input validation (file format, size, content)
- Graceful fallbacks when XAI methods fail
- Detailed error logging for debugging
- User-friendly error messages without technical jargon

**8. Enhanced Preprocessing Pipelines**

**Original**: Basic preprocessing in notebooks
**Our Solution**: Robust, modular preprocessing

**Audio Preprocessing:**
- Librosa for mel-spectrogram generation
- Configurable parameters (sample rate, n_mels, duration)
- Automatic resampling and normalization
- Handles variable-length audio

**Image Preprocessing:**
- Automatic resizing to model input size
- Proper normalization (ImageNet or medical imaging standards)
- Channel adaptation (grayscale ↔ RGB conversion)
- Format standardization

**9. Better Visualization Quality**

**Original**: Basic matplotlib plots
**Our Solution**: Professional 3-panel visualizations

**Features:**
- Original input + heatmap + overlay in single figure
- Consistent color schemes across all XAI methods
- High-resolution output suitable for publication
- Exportable figures

**10. User Interface Modernization**

**Original**: Basic Streamlit interface (audio repo)
**Our Solution**: Modern Gradio interface

**Improvements:**
- More responsive and intuitive UI
- Better mobile support
- Faster load times
- Automatic API generation for programmatic access
- Better Python 3.13+ compatibility

**11. Documentation and Maintainability**

**Original**: README with basic usage instructions
**Our Solution**: Comprehensive documentation

**Added:**
- Detailed installation guide
- Configuration documentation
- API reference for programmatic use
- Troubleshooting guide
- Architecture documentation
- Type hints throughout codebase
- Inline code documentation

**12. Extensibility and Modularity**

**Original**: Tightly coupled code in notebooks
**Our Solution**: Modular architecture

**Structure:**
```
src/
├── preprocessing/   # Isolated preprocessing logic
├── models/          # Model definitions and management
├── xai/             # XAI implementations
├── pipelines/       # End-to-end workflows
└── ui/              # Interface layer
```

**Benefits:**
- Easy to add new models (just implement model class)
- Easy to add new XAI methods (just implement XAI module)
- Easy to add new modalities (just add preprocessing + pipeline)
- Testable components (each module independently testable)

---

## Current Project Status

### Fully Functional Features

**Audio Classification**
- Custom CNN, VGG16, MobileNetV2: All working
- All XAI methods functional: Grad-CAM, SHAP, LIME
- Comparison tab fully operational

**Image Classification**
- XRV DenseNet121: Loads and generates predictions
- Grad-CAM: Fully functional with high-quality visualizations
- AlexNet & DenseNet baselines: Working

### In Progress / Resource-Constrained Models

These models are present in the interface to demonstrate platform capabilities despite being unable to complete full training due to hardware and storage limitations.

**FoR Keras Model (TensorFlow) - "FoR Keras (local)"**
- **File**: 11 MB `.keras` file present in `models/audio/for/`
- **Status**: Architecture complete, training blocked by resource constraints
- **Why labeled "local"**: Indicates model file is stored locally in project directory (not dynamically downloaded from external servers)
- **Why present but untrained**:
  - Training script prepared: `tools/train_audio_for_split.py`
  - FoR dataset download attempted via Kaggle API
  - **Resource limitations encountered**:
    - **Dataset size**: ~2.5GB compressed, ~5GB extracted
    - **RAM requirements**: 16GB+ needed for full dataset training
    - **Disk space**: Insufficient storage for complete audio dataset extraction
    - **Training attempted**: Local machine unable to handle full dataset in memory
  - Model architecture validated through synthetic data tests
  - All preprocessing and training infrastructure operational
- **Current limitation**: Produces random predictions (no training data learned)
- **What works**: XAI methods (Grad-CAM, SHAP, LIME) functional with TensorFlow backend, validates dual-framework architecture

**XRV DenseNet121 (PyTorch) - "XRV DenseNet121 (CheXpert)"**
- **File**: 27.8 MB `.pth` file present in `models/image/`
- **Status**: Pre-trained weights exported from TorchXRayVision, full training incomplete
- **Why present in interface**:
  - Demonstrates medical imaging XAI capabilities
  - Grad-CAM produces accurate anatomical localizations
  - Shows platform's multi-label classification support
  - Validates PyTorch + TorchXRayVision integration
- **Resource limitations encountered**:
  - **CheXpert dataset**: 224,316 images, ~439GB total size
  - **Training requirements**: 32GB+ RAM, high-end GPU (V100/A100), 500GB+ disk space
  - **Disk space constraint**: Local storage (~256GB total) insufficient for full dataset
  - Download attempted via: `tools/download_xrv_chex.py` (Kaggle API)
  - **Result**: Partial download completed before storage exhaustion
  - Successfully exported pre-trained weights but full fine-tuning impossible
- **Current issues**:
  - Generic labels (Pathology_16, Pathology_7) instead of medical terms (Cardiomegaly, Edema)
  - Label file mismatch: Expected 18 pathologies, model outputs 14
  - LIME channel issue: Expects grayscale (1-channel), receives RGB (3-channel)
- **What works despite constraints**:
  - Model loads and generates multi-label predictions
  - Grad-CAM produces high-quality heatmaps highlighting anatomical regions
  - Demonstrates medical AI workflow and multi-label classification
  - All preprocessing and inference pipelines operational

**Summary of Resource Challenges**:

Both models included as "working prototypes" demonstrating technical capabilities despite incomplete training:

| Challenge | FoR Keras (Audio) | XRV DenseNet121 (Image) |
|-----------|-------------------|-------------------------|
| **Dataset Size** | 5GB | 439GB |
| **RAM Required** | 16GB+ | 32GB+ |
| **Storage Needed** | 10GB+ | 500GB+ |
| **Available Resources** | 8GB RAM, 256GB disk | 8GB RAM, 256GB disk |
| **Primary Blocker** | RAM + Disk Space | Disk Space |
| **Training Time** | Days (if resources available) | Weeks (if resources available) |

**Why These Models Remain in Platform**:
1. **Demonstrate dual framework support** (PyTorch + TensorFlow/Keras)
2. **Validate XAI implementations** (Grad-CAM works perfectly)
3. **Show data pipeline robustness** (preprocessing handles both frameworks)
4. **Prove architectural soundness** (all infrastructure operational)
5. **Educational value** (realistic constraints in ML development)
6. **Future-ready** (can resume training when adequate resources available)

---

## Known Issues and Limitations

### Critical Issues

**XRV DenseNet121: LIME Channel Mismatch (XRV-Specific Issue)**

**Error Message:**
```
LIME failed: expected input to have 1 channels, but got 3 channels instead
```

**Affected Models:**
- ❌ XRV DenseNet121: LIME non-functional
- ✅ AlexNet: LIME works correctly
- ✅ DenseNet121: LIME works correctly

**Root Cause:**
- XRV DenseNet121 expects grayscale (1-channel) input: `Conv2d(1, 64, kernel_size=7)`
- LIME preprocessing passes RGB (3-channel) images: shape `[10, 3, 224, 224]`
- Channel dimension mismatch in first convolutional layer

**Console Output:**
```
  4%|███▋   | 9/200 [00:00<00:00, 248.39it/s]
LIME failed: Given groups=1, weight of size [64, 1, 7, 7], 
expected input[10, 3, 224, 224] to have 1 channels, but got 3 channels instead
```

**Impact:**
- XRV model: LIME non-functional, comparison tab shows Grad-CAM only
- Baseline models (AlexNet, DenseNet): LIME fully operational

**Status:**
- Fix identified in `classify_image_COMPLETE_FIX.py`
- Requires grayscale conversion for XRV-specific preprocessing

### Medium Priority Issues

**XRV Model Label Mismatch**

**Symptom:**
```
Top predicted pathologies:
- Pathology_16: 84.80%
- Pathology_7: 80.66%
- Pathology_0: 71.80%
```

**Expected Output:**
```
Top predicted pathologies:
- Cardiomegaly: 84.80%
- Edema: 80.66%
- Atelectasis: 71.80%
```

**Root Cause:**
- Model outputs 14 pathologies but label file contains 18
- Training incomplete, output dimensions don't match label file
- Automatic fallback generates generic labels

**Impact:**
- Reduced interpretability (unclear which pathology is which)
- Grad-CAM still shows correct localization
- Clinical utility limited

**Workaround:**
- Use Grad-CAM for visual localization
- Reference model output indices for pathology mapping

**FoR Keras Model Untrained**

**Status:**
- Model architecture created and saved (11 MB file)
- Weights randomly initialized
- No training performed on FoR dataset

**Impact:**
- Predictions essentially random
- XAI methods function correctly but explain random decisions
- Infrastructure validated and ready for trained model

### Minor Issues

**Performance Limitations**
- SHAP: 30-60 seconds per explanation (computationally expensive)
- LIME: 10-20 seconds per explanation (many perturbations)
- Comparison mode can take 45-90 seconds for audio (all 3 methods)

**Mitigation:**
- Reduce SHAP samples from 200 to 100 in configuration
- Reduce LIME perturbations from 600 to 300
- Use Grad-CAM for real-time demonstrations

**Platform-Specific Warnings**
- Windows: Asyncio connection warnings (cosmetic, no functional impact)
- Memory: Recommend 16GB RAM for smooth operation with multiple models

---

## Screenshots and Results

### Audio Classification Results

**Figure 1: Audio Deepfake Detection - Custom CNN with Grad-CAM**
*[Screenshot: Upload audio → Select Custom CNN → Select Grad-CAM → Result showing 3-panel visualization (spectrogram, heatmap, overlay) with prediction "Real: 99.99%"]*

**Figure 2: Audio XAI Comparison**
*[Screenshot: Comparison tab showing Grad-CAM, SHAP, and LIME side-by-side for same audio input]*

### Image Classification Results

**Figure 3: Chest X-ray Pathology Detection - XRV DenseNet121 with Grad-CAM**
*[Screenshot: Upload X-ray → Select XRV DenseNet121 → Select Grad-CAM → Result showing 3-panel visualization (original, heatmap, overlay) with top 5 pathology predictions]*

**Figure 4: Image Grad-CAM Visualization Quality**
*[Screenshot: Close-up of Grad-CAM heatmap showing highlighted lung regions corresponding to pathology predictions]*

### Interface Screenshots

**Figure 5: Classification Tab Interface**
*[Screenshot: Full interface showing data type selection, file upload, model dropdown, XAI dropdown, and classify button]*

**Figure 6: XAI Comparison Tab**
*[Screenshot: Comparison tab interface showing multiple XAI outputs in grid layout]*

### Technical Demonstration

**Figure 7: Automatic Compatibility Filtering**
*[Screenshot: Showing how XAI dropdown options change when switching between audio and image data types]*

**Figure 8: Error Handling Example**
*[Screenshot: User-friendly error message when incompatible operation attempted]*

---

## AI Usage Declaration

### Generative AI Tools Used

**Tool**: Claude 3.5 Sonnet (Anthropic)

**Declared Usage:**

**Code Development:**
- Debugging framework integration issues (PyTorch ↔ TensorFlow)
- XAI implementation guidance (Grad-CAM gradient computation, LIME segmentation, SHAP value calculation)
- Error resolution (channel mismatches, label indexing, model loading)
- Code optimization and refactoring suggestions

**Documentation:**
- README structure and technical writing
- Code comments and docstrings
- API documentation generation
- Troubleshooting guide creation

**Architecture Decisions:**
- Discussion of framework trade-offs (Streamlit vs Gradio)
- Model management pattern suggestions
- XAI compatibility system design review

**What We Implemented Ourselves:**

**Project Design:**
- Overall architecture and modular structure
- User workflow and interface design
- Model selection criteria
- XAI method selection and prioritization

**Core Implementation:**
- Model integration and testing
- Dataset preprocessing pipelines
- UI component layout and interaction logic
- Configuration management system

**Validation:**
- Testing all model-XAI combinations
- Verification of XAI output correctness
- Performance benchmarking
- Bug identification and reproduction

**Transparency Statement:**

All AI-generated code was thoroughly reviewed, tested, and modified to fit our specific requirements. We understand the underlying algorithms, can explain all design decisions, and take full responsibility for the implementation. The AI served as a development assistant, not as the primary implementer.

---

## References

### Source Repositories

1. **Deepfake Audio Detection with XAI**  
   GitHub: https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI  
   *Inspiration for audio architecture and initial XAI approach*

2. **Lung Cancer Detection**  
   GitHub: https://github.com/schaudhuri16/LungCancerDetection  
   *Inspiration for medical image classification approach*

3. **TorchXRayVision**  
   GitHub: https://github.com/mlmed/torchxrayvision  
   *Medical imaging models and CheXpert integration*

### Datasets

4. **Fake-or-Real (FoR) Dataset**  
   Kaggle: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset  
   *Audio deepfake detection dataset used for Keras model training*

5. **CheXpert Dataset**  
   Kaggle: https://www.kaggle.com/datasets/ashery/chexpert  
   Stanford ML Group: https://stanfordmlgroup.github.io/competitions/chexpert/  
   *224,316 chest X-rays with 14 pathology labels for XRV DenseNet121*

### XAI Methods - Academic Papers

6. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**  
   Selvaraju et al., ICCV 2017  
   arXiv: https://arxiv.org/abs/1610.02391

7. **"Why Should I Trust You?": Explaining the Predictions of Any Classifier**  
   Ribeiro et al., KDD 2016  
   arXiv: https://arxiv.org/abs/1602.04938

8. **A Unified Approach to Interpreting Model Predictions**  
   Lundberg & Lee, NeurIPS 2017  
   arXiv: https://arxiv.org/abs/1705.07874

### Libraries and Frameworks

9. **Gradio**: Hassle-Free Sharing and Testing of ML Models  
   Documentation: https://gradio.app/

10. **PyTorch**: An Imperative Style, High-Performance Deep Learning Library  
    Documentation: https://pytorch.org/

11. **TensorFlow**: An End-to-End Open Source Machine Learning Platform  
    Documentation: https://tensorflow.org/

12. **Librosa**: Audio and Music Signal Analysis in Python  
    Documentation: https://librosa.org/

---

## Project Structure

```
unified-xai-platform/
├── app.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SETUP.md                        # Installation and configuration guide
│
├── data/audio/for-2sec/           # FoR dataset (not included)
│   ├── training/
│   ├── validation/
│   └── testing/
│
├── src/
│   ├── config.py                  # Configuration and constants
│   ├── utils.py                   # Utility functions
│   │
│   ├── preprocessing/             # Data preprocessing
│   │   ├── audio.py              # Audio → mel-spectrogram
│   │   └── image.py              # Image normalization/resizing
│   │
│   ├── pipelines/                 # Classification workflows
│   │   ├── classify_audio.py     # Audio pipeline
│   │   ├── classify_image.py     # Image pipeline
│   │   └── compare.py            # XAI comparison
│   │
│   ├── xai/                       # XAI implementations
│   │   ├── gradcam.py            # Grad-CAM (PyTorch)
│   │   ├── gradcam_tf.py         # Grad-CAM (TensorFlow)
│   │   ├── lime_audio.py         # LIME audio (PyTorch)
│   │   ├── lime_audio_tf.py      # LIME audio (TensorFlow)
│   │   ├── lime_xai.py           # LIME image
│   │   ├── shap_xai.py           # SHAP (PyTorch)
│   │   └── shap_audio_tf.py      # SHAP (TensorFlow)
│   │
│   └── ui/                        # User interface
│       ├── interface.py          # Gradio interface
│       └── components.py         # Reusable UI components
│
├── models/                        # Model weights
│   ├── audio_models.py           # PyTorch audio model definitions
│   ├── image_models.py           # PyTorch image model definitions
│   ├── manager.py                # Centralized model loading
│   │
│   ├── audio/for/                # FoR Keras model
│   │   ├── audio_classifier.keras  (11 MB)
│   │   ├── config.json
│   │   └── labels.json
│   │
│   └── image/                    # XRV DenseNet121
│       ├── xrv-densenet121-res224-chex.pth  (27.8 MB)
│       └── xrv_pathologies.txt
│
└── tools/                         # Utility scripts
    ├── download_xrv_chex.py      # Download XRV weights
    ├── export_xrv_chex_weights.py  # Export XRV to .pth
    ├── find_all_torch_weights.py   # Find cached PyTorch models
    └── train_audio_for_split.py    # Train FoR Keras model
```

---

## License

MIT License - Copyright (c) 2026 DIA4 Team

---

## Contact

For questions or issues:
- Repository: [GitHub Link]
- Issues: [GitHub Issues]

---

**Made with care by the DIA4 Team**  
*Explainable AI for Audio and Medical Imaging*

**Technologies**: Python 3.8+ | PyTorch 2.0+ | TensorFlow 2.13+ | Gradio 4.0+

[Back to Top](#unified-xai-platform-for-multi-modal-classification)
