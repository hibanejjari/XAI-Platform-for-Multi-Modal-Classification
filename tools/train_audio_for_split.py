import os
import json
import numpy as np
import librosa

from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# Paths
# -----------------------------
# -----------------------------
# Paths (ABSOLUTE, FIXED)
# -----------------------------
BASE = r"C:\Users\HIBAN\Downloads\unified-xai-platform (2)\unified-xai-platform\data\audio\for-2sec"

TRAIN_DIR = os.path.join(BASE, "training")
VAL_DIR   = os.path.join(BASE, "validation")
TEST_DIR  = os.path.join(BASE, "testing")

OUT_DIR = r"C:\Users\HIBAN\Downloads\unified-xai-platform (2)\unified-xai-platform\models\audio\for"
os.makedirs(OUT_DIR, exist_ok=True)

print("TRAIN_DIR =", TRAIN_DIR)
print("Exists:", os.path.isdir(TRAIN_DIR))
print("Has real:", os.path.isdir(os.path.join(TRAIN_DIR, "Real")))
print("Has fake:", os.path.isdir(os.path.join(TRAIN_DIR, "Fake")))


# -----------------------------
# Audio/Spectrogram config
# -----------------------------
SR = 16000
DURATION = 2.0           # for-2sec already, but keep for safety
N_MELS = 128
HOP_LENGTH = 512

LABELS = {"Real": 0, "Fake": 1}

# Optional: speed up by limiting files per class per split
MAX_PER_CLASS = None  # e.g. 5000

def list_audio_files(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
                out.append(os.path.join(root, f))
    return out

def load_fixed_length(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    target = int(SR * DURATION)
    if len(y) > target:
        y = y[:target]
    elif len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    return y

def to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to [0,1]
    mel_db -= mel_db.min()
    mel_db /= (mel_db.max() + 1e-8)
    return mel_db.astype(np.float32)

def build_split(split_dir):
    X, y = [], []
    for cls in ["Real", "Fake"]:
        cls_dir = os.path.join(split_dir, cls)
        files = sorted(list_audio_files(cls_dir))
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]
        for p in files:
            wav = load_fixed_length(p)
            mel = to_mel(wav)
            X.append(mel)
            y.append(LABELS[cls])

    X = np.stack(X, axis=0)      # (N, n_mels, time)
    y = np.array(y, dtype=np.int64)
    X = X[..., None]             # (N, n_mels, time, 1)
    return X, y

def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.isdir(os.path.join(d, "real")) or not os.path.isdir(os.path.join(d, "fake")):
            raise RuntimeError(f"Expected real/ and fake/ under: {d}")

    print("Loading training split...")
    X_train, y_train = build_split(TRAIN_DIR)
    print("Loading validation split...")
    X_val, y_val = build_split(VAL_DIR)
    print("Loading testing split...")
    X_test, y_test = build_split(TEST_DIR)

    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    model = build_model(X_train.shape[1:])
    cb = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "audio_classifier.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=cb
    )

    print("Evaluating on test...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", float(test_acc))

    # Save labels + config
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"0": "real", "1": "fake"}, f, indent=2)

    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"sr": SR, "duration": DURATION, "n_mels": N_MELS, "hop_length": HOP_LENGTH},
            f, indent=2
        )

    print("Saved model to:", os.path.join(OUT_DIR, "audio_classifier.keras"))

if __name__ == "__main__":
    main()
