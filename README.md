<<<<<<< HEAD
# 🔬 Unified XAI Platform for Multi-Modal Classification

An interactive platform that integrates **Audio Deepfake Detection** and **Lung Cancer Detection** with comprehensive Explainable AI (XAI) capabilities.

## 📋 Overview

This project combines two state-of-the-art classification systems into a single, user-friendly interface:

### 🎵 Audio: Deepfake Detection
- **Models:** Custom CNN, MobileNet, VGG16
- **Dataset:** Fake-or-Real (FoR)
- **XAI Methods:** Grad-CAM, SHAP
- **Task:** Distinguish between real and synthetic audio

### 🫁 Image: Lung Cancer Detection
- **Models:** AlexNet, DenseNet
- **Dataset:** CheXpert chest X-rays
- **XAI Methods:** Grad-CAM, LIME
- **Task:** Identify malignant tumors in chest X-rays

## ✨ Features

- **Multi-Modal Support:** Process both audio (.wav) and image files
- **Multiple Models:** Choose from various pre-trained architectures
- **XAI Integration:** Understand model decisions with visual explanations
- **Automatic Compatibility:** Only relevant XAI methods shown for each data type
- **Side-by-Side Comparison:** Compare different XAI techniques simultaneously
- **Interactive UI:** Built with Gradio for easy interaction

## 🏗️ Project Structure

```
unified-xai-platform/
│
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── utils.py               # Utility functions
│   │
│   ├── preprocessing/         # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── audio.py          # Audio preprocessing (mel-spectrograms)
│   │   └── image.py          # Image preprocessing
│   │
│   ├── models/                # Model definitions and management
│   │   ├── __init__.py
│   │   ├── audio_models.py   # Audio classification models
│   │   ├── image_models.py   # Image classification models
│   │   └── manager.py        # Model loading and caching
│   │
│   ├── xai/                   # Explainable AI methods
│   │   ├── __init__.py
│   │   ├── gradcam.py        # Grad-CAM implementation
│   │   ├── lime_xai.py       # LIME for images
│   │   └── shap_xai.py       # SHAP for audio
│   │
│   ├── pipelines/             # End-to-end workflows
│   │   ├── __init__.py
│   │   ├── classify_audio.py # Audio classification pipeline
│   │   ├── classify_image.py # Image classification pipeline
│   │   └── compare.py        # XAI comparison workflow
│   │
│   └── ui/                    # User interface
│       ├── __init__.py
│       ├── components.py     # Reusable UI components
│       └── interface.py      # Main Gradio interface
│
├── models/                    # Trained model weights (add your .pt/.pth files)
│   ├── audio/
│   └── image/
│
└── examples/                  # Example input files
    ├── audio.wav
    └── xray.png
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or extract the project:**
```bash
cd unified-xai-platform
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Add your trained models (optional):**
   - Place audio model weights in `models/audio/`
   - Place image model weights in `models/image/`

## 💻 Usage

### Running the Application

```bash
python app.py
```

The Gradio interface will launch in your default web browser at `http://localhost:7860`

### Using the Interface

#### Classification Tab
1. Select data type (Audio or Image)
2. Upload your file
3. Choose a classification model
4. Select an XAI method
5. Click "Classify & Explain"
6. View prediction results and XAI visualization

#### Comparison Tab
1. Select data type (Audio or Image)
2. Upload your file
3. Choose a classification model
4. Click "Compare XAI Methods"
5. View side-by-side comparison of all available XAI techniques

## 🧠 XAI Methods

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Available for:** Audio & Image
- **What it shows:** Heatmap highlighting regions most important for prediction
- **Use case:** Identify which parts of input influenced the model's decision

### LIME (Local Interpretable Model-agnostic Explanations)
- **Available for:** Image only
- **What it shows:** Superpixel-based explanation of important regions
- **Use case:** Understand local decision boundaries

### SHAP (SHapley Additive exPlanations)
- **Available for:** Audio only
- **What it shows:** Feature importance based on game theory
- **Use case:** Quantify contribution of different frequency bands

## 🔧 Technical Details

### Audio Processing
- Converts audio files to mel-spectrograms (128x128)
- Sample rate: 22,050 Hz
- Duration: 3 seconds
- Normalization: Z-score normalization

### Image Processing
- Resizes images to 224x224
- RGB color space
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Architecture Adaptations
- **Custom CNN:** Designed specifically for audio spectrograms
- **MobileNet/VGG16:** Adapted for audio (3-channel replication)
- **AlexNet/DenseNet:** Pre-trained on ImageNet, fine-tuned for medical imaging

## 📊 Model Performance Notes

The provided models are **demonstration versions** and may not have optimal accuracy. For production use:

1. Train models on larger, domain-specific datasets
2. Implement proper data augmentation
3. Perform hyperparameter tuning
4. Validate on held-out test sets

## 🤝 Contributing

This project is designed for educational purposes. To extend it:

1. **Add new models:** Implement in `src/models/`
2. **Add new XAI methods:** Implement in `src/xai/`
3. **Support new data types:** Add preprocessing in `src/preprocessing/`
4. **Enhance UI:** Modify `src/ui/interface.py`

## 📝 Credits

This platform integrates concepts from:
- **Deepfake Audio Detection:** [Original Repository]
- **Lung Cancer Detection:** [Original Repository]

### Libraries Used
- **Gradio:** Web UI framework
- **PyTorch:** Deep learning framework
- **Librosa:** Audio processing
- **OpenCV:** Image processing
- **LIME:** Explainable AI
- **SHAP:** Explainable AI

## 📄 License

This project is for educational purposes. Please respect the licenses of individual components and datasets.

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**"CUDA out of memory"**
- The code automatically falls back to CPU if CUDA is unavailable
- Reduce batch size if processing multiple files

**"Model not found"**
- Models are initialized randomly if weights are not provided
- Add your trained model weights to the `models/` directory

**"LIME/SHAP not available"**
- Install optional dependencies: `pip install lime shap scikit-image`

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Examine the example files in `examples/`

## 🎯 Future Enhancements

- [ ] Add more model architectures (ResNet, EfficientNet, Transformers)
- [ ] Support additional file formats (MP3, FLAC, DICOM)
- [ ] Implement batch processing
- [ ] Add model training capabilities
- [ ] Export results to PDF/CSV
- [ ] Add confidence calibration
- [ ] Implement attention mechanisms
- [ ] Add more XAI methods (Integrated Gradients, Occlusion)

---

**Built with ❤️ for Explainable AI Research**
=======
# XAI-Platform-for-Multi-Modal-Classification
Integrating Audio Deepfake Detection and Chest X-ray Analysis with Explainability
>>>>>>> 7def7da9370e4767765b7b142e74f9077c6a5f5f
