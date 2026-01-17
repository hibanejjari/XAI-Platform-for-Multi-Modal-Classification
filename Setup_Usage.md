
## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_REPO/unified-xai-platform.git
cd unified-xai-platform

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

**Requirements**: Python 3.8-3.13, 8GB+ RAM

### Configuration

Edit `src/config.py` to customize models, paths, and XAI compatibility:

```python
# Model paths
AUDIO_KERAS_MODEL_PATH = "models/audio/for/audio_classifier.keras"
XRV_IMAGE_WEIGHTS_PATH = "models/image/xrv-densenet121-res224-chex.pth"

# XAI compatibility mapping
XAI_COMPATIBILITY = {
    "Audio": {
        "Custom CNN": ["Grad-CAM", "SHAP", "LIME"],
        "VGG16": ["Grad-CAM", "SHAP", "LIME"],
        # ...
    },
    "Image": {
        "XRV DenseNet121 (CheXpert)": ["Grad-CAM", "LIME"],
        # ...
    }
}
```

---

## 💻 Usage

### Classification Tab
1. Select **Audio** or **Image**
2. Upload file (`.wav` or chest X-ray image)
3. Choose model (e.g., VGG16 for audio, XRV DenseNet121 for images)
4. Select XAI method
5. Click **Classify & Explain**

**Working Example**:
```
Audio: Custom CNN + Grad-CAM ✅
→ Prediction: Real (99.99%), 3-panel visualization
```

### Comparison Tab
1. Upload file
2. Select model
3. Click **Compare XAI Methods**
4. View all compatible methods side-by-side

**Working Example**:
```
Audio: VGG16 → Grad-CAM + SHAP + LIME ✅
Image: XRV DenseNet121 → Grad-CAM only ⚠️ (LIME broken)
```

---
