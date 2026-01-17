# Unified XAI Platform for Multi-Modal Classification

**Integrating Audio Deepfake Detection and Chest X-ray Analysis with Explainability**
---

## Team ( ESILV A5 : DIA4 )

**Lisa NACCACHE** • **Hiba NEJJARI** • **Neil MAHCER** • **Wendy DUONG** • **Cyprien MOUTON**

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Status](#project-status)
- [Technical Details](#technical-details)
- [Known Issues](#known-issues)
- [AI Usage Declaration](#ai-usage-declaration)
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



### Key Improvements 
- ✅ **Unified Interface**: Single Gradio app (migrated from Streamlit for Python 3.13 compatibility)
- ✅ **Dual Framework**: PyTorch + TensorFlow/Keras support
- ✅ **Auto-Compatibility**: XAI methods filtered by input type
- ✅ **Comparison Mode**: Side-by-side XAI visualization

---

## 🎥 Demo

**[📺 Watch Demo Video](YOUR_VIDEO_LINK_HERE)**

---

## ✨ Features

| Feature | Audio | Image | Status |
|---------|-------|-------|--------|
| **Classification** | ✅ | ✅ | Working |
| **Grad-CAM** | ✅ | ✅ | Fully functional |
| **SHAP** | ✅ | ⏳ | Audio only |
| **LIME** | ✅ | ❌ | Image has channel error |
| **Comparison Tab** | ✅ | ⚠️ | Grad-CAM only for images |

---


## 🚧 Project Status

### ✅ Working Features

**Audio Classification** (Fully Functional)
- PyTorch models: Custom CNN, VGG16, MobileNetV2 ✅
- All XAI methods working: Grad-CAM, SHAP, LIME ✅
- Comparison tab functional ✅

**Image Classification** (Partial)
- XRV DenseNet121 loads and predicts ✅
- Grad-CAM works perfectly ✅
- Shows predictions (generic labels) ⚠️

### ⚠️ In Progress

**FoR Keras Model** (Audio - TensorFlow)
- File: `audio_classifier.keras` (11 MB)
- Status: Architecture ready, **not trained yet**
- Issue: Random predictions (training required)

**XRV DenseNet121** (Image - PyTorch)  
- File: `xrv-densenet121-res224-chex.pth` (27.8 MB)
- Status: Weights exported, **training incomplete**
- Issues:
  - Generic labels (`Pathology_16` instead of `Cardiomegaly`)
  - LIME fails with channel mismatch

---

## 📊 Technical Details

### Design Decisions

**1. Streamlit → Gradio Migration**
- Better Python 3.13 support
- Built-in ML components
- Faster development

**2. Dual Framework Architecture**
```
PyTorch Models → PyTorch XAI (gradcam.py, shap_xai.py, lime_audio.py)
TensorFlow Models → TF XAI (gradcam_tf.py, shap_audio_tf.py, lime_audio_tf.py)
```

**3. Model Management**
- Centralized `ModelManager` with caching
- Lazy loading for memory efficiency
- Automatic framework detection

### XAI Implementation

| Method | Framework | Speed | Accuracy | Status |
|--------|-----------|-------|----------|--------|
| **Grad-CAM** | PyTorch + TF | ⚡ Fast (1-4s) | ⭐⭐⭐ | ✅ All models |
| **LIME** | PyTorch + TF | 🐌 Slow (10-20s) | ⭐⭐ | ✅ Audio, ❌ Image |
| **SHAP** | PyTorch + TF | 🐢 Very Slow (30-60s) | ⭐⭐⭐ | ✅ Audio only |

### Improvements Over Original Repos

1. ✅ **Unified Platform**: Single interface vs. separate projects
2. ✅ **Auto-Compatibility**: Dynamic XAI filtering
3. ✅ **Dual Framework**: PyTorch + TensorFlow integration
4. ✅ **Better UX**: Modern Gradio UI with comparison mode
5. ✅ **Error Handling**: Comprehensive fallbacks and validation
6. ✅ **Model Caching**: Faster repeated inference

---

## ⚠️ Known Issues

### 🔴 Critical

**Image LIME Channel Error**
```
LIME failed: expected input to have 1 channels, but got 3 channels instead
```
- **Cause**: XRV model expects grayscale, LIME passes RGB
- **Impact**: Image LIME unusable, comparison limited
- **Fix**: In progress (`classify_image_COMPLETE_FIX.py`)

### 🟡 Medium

**Generic Pathology Labels**
```
Top predicted pathologies:
- Pathology_16: 84.80%  ❌ Should be "Cardiomegaly"
- Pathology_7: 80.66%   ❌ Should be "Edema"
```
- **Cause**: Model training incomplete, label mismatch
- **Impact**: Unclear which pathology is which
- **Workaround**: Use Grad-CAM for localization

**FoR Keras Untrained**
- Model file present but weights random
- Predictions unreliable
- Requires FoR dataset training

### 🟢 Minor

- Windows asyncio warnings (cosmetic)
- SHAP/LIME slow (reduce samples for speed)

### Fixes Provided

See `EMERGENCY_PATCH.md` for detailed fixes:
1. `classify_audio_CORRECTED.py` - TensorFlow XAI integration
2. `classify_image_COMPLETE_FIX.py` - Label fix + LIME prep
3. `gradcam_tf_FIXED.py` - Keras model building fix
4. `compare_FIXED.py` - Compatibility filtering

---

## 🤖 AI Usage Declaration

### Generative AI Tools Used

**Tool**: Claude 3.5 Sonnet (Anthropic)

**Purpose**:
- ✅ Code debugging and error resolution
- ✅ XAI implementation guidance (Grad-CAM, LIME, SHAP)
- ✅ Framework integration (PyTorch ↔ TensorFlow)
- ✅ Documentation writing
- ✅ Bug fix generation (channel mismatch, label errors)

**What We Did Ourselves**:
- ✅ Project architecture design
- ✅ Model selection and integration decisions
- ✅ UI/UX design and workflow
- ✅ Testing and validation
- ✅ Dataset understanding and preprocessing

**Transparency Statement**: 
All AI-generated code was reviewed, tested, and adapted to our specific requirements. We understand the implementation and can explain all design decisions.

---

## 📚 References

### Original Repositories
1. [Deepfake Audio Detector with XAI](https://github.com/source-audio-repo)
2. [Lung Cancer Detection](https://github.com/source-image-repo)

### Datasets
- **FoR (Fake-or-Real)**: Audio deepfake dataset
- **CheXpert**: 224,316 chest X-rays with 14 pathology labels

### XAI Papers
- **Grad-CAM**: Selvaraju et al. (ICCV 2017) - [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- **LIME**: Ribeiro et al. (KDD 2016) - [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
- **SHAP**: Lundberg & Lee (NeurIPS 2017) - [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

### Libraries
- **Gradio**: Web UI framework
- **TorchXRayVision**: Medical imaging models
- **PyTorch** + **TensorFlow**: Deep learning frameworks

---

## 📁 Project Structure

```
unified-xai-platform/
├── app.py                    # Entry point
├── requirements.txt
├── src/
│   ├── config.py            # ⚙️ Configuration
│   ├── preprocessing/       # Audio/image preprocessing
│   ├── pipelines/           # Classification + XAI workflows
│   ├── xai/                 # Grad-CAM, LIME, SHAP (PyTorch + TF)
│   └── ui/                  # Gradio interface
├── models/
│   ├── audio/for/           # FoR Keras (11 MB, untrained)
│   ├── image/               # XRV DenseNet121 (27.8 MB, partial)
│   ├── audio_models.py      # PyTorch audio models
│   ├── image_models.py      # PyTorch image models
│   └── manager.py           # Model loading
└── tools/                   # Model generation scripts
```

---

## 🎯 Future Work

**Immediate**:
- [ ] Train FoR Keras on FoR dataset
- [ ] Complete XRV DenseNet121 training
- [ ] Fix image LIME channel issue

**Planned**:
- [ ] Add more XAI methods (Integrated Gradients, SmoothGrad)
- [ ] Batch processing mode
- [ ] Export to PDF reports
- [ ] Docker deployment

---

## 📄 License

MIT License - Copyright (c) 2026 DIA4 Team

---

<div align="center">


- **Datasets**  :
- **Fake-or-Real (FoR)**: Audio deepfake detection dataset used for Keras model training  
- **CheXpert**: Chest X-ray pathology detection dataset (224,316 images, 14 pathology labels) used with XRV DenseNet121
- 
- **Development & Sources**  
- Deepfake Audio Detection with XAI (GitHub): https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI  
- Lung Cancer Detection (GitHub): https://github.com/schaudhuri16/LungCancerDetection  
- TorchXRayVision: https://github.com/mlmed/torchxrayvision  
- FoR Dataset (Kaggle): https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset  
- CheXpert Dataset (Stanford ML Group): https://www.kaggle.com/datasets/ashery/chexpert
**Made with ❤️ by DIA4**

*Explainable AI for Audio & Medical Imaging*

[![Python](https://img.shields.io/badge/Python-3.8--3.13-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)

[⬆ Back to Top](#-unified-xai-platform-for-multi-modal-classification)

</div>
