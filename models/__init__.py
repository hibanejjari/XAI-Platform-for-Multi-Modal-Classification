"""Models package for audio and image classification"""

from .audio_models import AudioCNN, get_audio_mobilenet, get_audio_vgg16, get_audio_target_layer
from .image_models import get_image_alexnet, get_image_densenet, get_image_target_layer
from .manager import ModelManager, model_manager
