"""
Audio preprocessing module
Handles conversion of audio files to mel-spectrograms
"""

import numpy as np
import librosa
import cv2
from src.config import AUDIO_SAMPLE_RATE, AUDIO_N_MELS, AUDIO_DURATION

def preprocess_audio(audio_path: str,
                     sr: int = AUDIO_SAMPLE_RATE,
                     n_mels: int = AUDIO_N_MELS,
                     duration: int = AUDIO_DURATION) -> np.ndarray:
    """
    Convert audio file to mel-spectrogram matching FoR training:
    - sr=16000
    - duration=2s (pad/trim)
    - n_mels=128
    - hop_length=512
    - normalize to [0,1]
    Returns: (128, T) float32
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize to [0,1] like your training script
    mel_db -= mel_db.min()
    mel_db /= (mel_db.max() + 1e-8)

    return mel_db.astype(np.float32)
