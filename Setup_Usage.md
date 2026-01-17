## How to set up 

### Installation

```bash
# Clone repository
git clone https://github.com/hibanejjari/XAI-Platform-for-Multi-Modal-Classification.git
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

## User guide to the application

### Classification Tab
1. Select **Audio** or **Image**
2. Upload or drag/drop file (`.wav` or chest X-ray image)
3. Choose model from drop down menu (like VGG16 for audio, DensNet for images)
4. Select XAI method from drop down
5. Click **Classify & Explain**

**Working Example**:

Audio: Custom CNN + Grad-CAM 
<img width="1532" height="760" alt="Capture d&#39;écran 2026-01-17 144146" src="https://github.com/user-attachments/assets/3d86dfef-afe9-49a0-aecd-b1e7ab1ec2bb" />


### Comparison Tab
1. Select **Audio** or **Image** upload or drag/drop file (`.wav` or chest X-ray image)
2. Select model from drop down
3. Click **Compare XAI Methods**
4. View all detailed compatible methods by clicking underneath the image


**Working Example**:

Audio: VGG16 → Grad-CAM + SHAP + LIME 

<img width="1702" height="759" alt="image" src="https://github.com/user-attachments/assets/3be4c875-2ff0-4ad6-8bad-5cb4b1441a81" />

---
