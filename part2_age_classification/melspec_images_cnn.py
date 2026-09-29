"""
Q2 (ii) - 2D CNN directly on the Mel-spectrogram IMAGE
--------------------------------------------------------
Step 1: render each clip's mel-spectrogram as a plain image
        (no axes, no colour-bar, no title - just the picture),
        saved as e.g. mel_images/<client_id>_<clip_id>.png
Step 2: train a 2D CNN (image classifier) on these images to predict age,
        using a speaker-independent (GroupShuffleSplit) train/test split.

Usage:
    python melspec_images_cnn.py --cv_root /path/to/hi --out_dir mel_images --stage images
    python melspec_images_cnn.py --features_dir features --img_dir mel_images --stage train
"""

import argparse
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from split_utils import balanced_sample, speaker_independent_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

SR = 16000
IMG_SIZE = (128, 128)
AGE_CLASSES = ["teens", "twenties", "thirties", "fourties"]


# ------------------------------------------------------------------ #
# Stage 1: image rendering, colour-bar & axes removed as required
# ------------------------------------------------------------------ #
def save_melspec_image(wav_path, out_png, sr=SR):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = plt.axes([0, 0, 1, 1])   # fill the whole figure - no margins
    ax.set_axis_off()
    librosa.display.specshow(mel_db, sr=sr, cmap="magma", ax=ax)
    fig.savefig(out_png, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def build_images(cv_root, tsv_name, out_dir, cap_per_class=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(cv_root, tsv_name), sep="\t")
    df = df[df["age"].isin(AGE_CLASSES)].reset_index(drop=True)
    df = balanced_sample(df, "age", cap_per_class)

    rows = []
    for _, row in df.iterrows():
        wav_path = os.path.join(cv_root, "clips", row["path"])
        if not os.path.exists(wav_path):
            continue
        out_png = os.path.join(out_dir, f"{row['client_id']}_{os.path.basename(row['path'])}.png")
        try:
            save_melspec_image(wav_path, out_png)
        except Exception as e:
            print("skip", wav_path, e)
            continue
        rows.append({"image": out_png, "client_id": row["client_id"], "age": row["age"]})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "image_meta.csv"), index=False)
    print(f"Saved {len(rows)} mel-spectrogram images -> {out_dir}")


# ------------------------------------------------------------------ #
# Stage 2: 2D CNN classifier
# ------------------------------------------------------------------ #
def load_image_array(path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    return img_to_array(img) / 255.0


def build_cnn(input_shape, n_classes):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_cnn(img_dir, epochs=25, out_dir="."):
    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        raise SystemExit(
            f"TensorFlow is required for the 2D CNN stage but failed to import "
            f"({type(e).__name__}: {e}).\n"
            f"On Apple Silicon Macs (M1/M2/M3), plain 'pip install tensorflow' does "
            f"not work - install instead with:\n"
            f"    pip install tensorflow-macos tensorflow-metal\n"
            f"On Intel Macs / Linux / Windows, 'pip install tensorflow' should work; "
            f"check the error above for the real cause if it still fails."
        )
    meta = pd.read_csv(os.path.join(img_dir, "image_meta.csv"))
    le = LabelEncoder()
    y = le.fit_transform(meta["age"])

    idx_train, idx_test = speaker_independent_split(y, meta["client_id"].values)

    print("Loading images into memory ...")
    X = np.stack([load_image_array(p) for p in meta["image"]])
    Xtr, Xte = X[idx_train], X[idx_test]
    ytr, yte = y[idx_train], y[idx_test]

    model = build_cnn(Xtr.shape[1:], len(le.classes_))
    history = model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs,
                         batch_size=32)

    y_pred = np.argmax(model.predict(Xte), axis=1)
    acc = accuracy_score(yte, y_pred)
    print(f"\n2D-CNN test accuracy: {acc:.4f}")
    report = classification_report(yte, y_pred,
                                    labels=range(len(le.classes_)),
                                    target_names=le.classes_,
                                    zero_division=0)
    print(report)

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("2D CNN on Mel-spectrogram images")
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, "cnn_training_curve.png")
    plt.savefig(curve_path, dpi=150)

    with open(os.path.join(out_dir, "cnn_results.txt"), "w") as f:
        f.write(f"2D-CNN test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Saved {curve_path} and cnn_results.txt")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["images", "train"], required=True)
    ap.add_argument("--cv_root")
    ap.add_argument("--tsv_name", default="validated.tsv")
    ap.add_argument("--out_dir", default="mel_images")
    ap.add_argument("--img_dir", default="mel_images")
    ap.add_argument("--cap_per_class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--results_dir", default=".",
                     help="where to save cnn_training_curve.png / cnn_results.txt")
    args = ap.parse_args()

    if args.stage == "images":
        build_images(args.cv_root, args.tsv_name, args.out_dir, args.cap_per_class)
    else:
        train_cnn(args.img_dir, args.epochs, args.results_dir)
