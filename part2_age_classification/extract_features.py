"""
Q2 - Feature extraction for age classification on Common Voice Hindi
---------------------------------------------------------------------
Produces THREE feature sets per clip, saved as .npz / .csv:

  1. GeMAPS  (62 features)      - via openSMILE
  2. eGeMAPS (88 features)      - via openSMILE
  3. Custom "180" set           - MFCC(40) + Mel-spectrogram(128) + Chroma(12)
                                   each averaged over time -> 40+128+12 = 180

Expected Common Voice directory layout (after unzipping the dataset):

    cv-corpus-26.0-2026-06-12/hi/
        clips/*.mp3
        validated.tsv   (has client_id, path, age, gender, ...)

Usage:
    pip install opensmile librosa pandas tqdm soundfile
    python extract_features.py --cv_root /path/to/cv-corpus-26.0-2026-06-12/hi \
                                --out_dir features/
"""

import argparse
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from split_utils import balanced_sample

AGE_CLASSES = ["teens", "twenties", "thirties", "fourties"]  # per assignment
SR = 16000


def load_metadata(cv_root):
    tsv_path = os.path.join(cv_root, "validated.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["age"].isin(AGE_CLASSES)].reset_index(drop=True)
    df["age"] = df["age"].replace({"fourties": "forties"})  # normalise spelling
    return df


# ------------------------------------------------------------------ #
# 1 & 2: openSMILE GeMAPS (62) / eGeMAPS (88)
# ------------------------------------------------------------------ #
def build_opensmile_extractors():
    import opensmile
    gemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )   # 62-dim functionals
    egemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )   # 88-dim functionals
    return gemaps, egemaps


def opensmile_features(wav_path, gemaps, egemaps):
    g = gemaps.process_file(wav_path).values.flatten()
    e = egemaps.process_file(wav_path).values.flatten()
    return g, e


# ------------------------------------------------------------------ #
# 3: Custom 180-d set: MFCC(40) + Mel-spectrogram(128) + Chroma(12)
# ------------------------------------------------------------------ #
def custom_180_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    if len(y) < 400:
        y = np.pad(y, (0, 400 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)          # (40, T)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # (128, T)
    mel_db = librosa.power_to_db(mel)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)  # (12, T)

    feat = np.concatenate([
        mfcc.mean(axis=1),      # 40
        mel_db.mean(axis=1),    # 128
        chroma.mean(axis=1),    # 12
    ])   # -> 180
    return feat


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_root", required=True)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--cap_per_class", type=int, default=None,
                     help="max clips per age class (balanced, speaker-spread sample)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_metadata(args.cv_root)
    df = balanced_sample(df, "age", args.cap_per_class)
    print("Clips per class:\n", df["age"].value_counts().to_string())

    gemaps, egemaps = build_opensmile_extractors()

    rows = []
    X62, X88, X180 = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = os.path.join(args.cv_root, "clips", row["path"])
        if not os.path.exists(wav_path):
            continue
        try:
            g, e = opensmile_features(wav_path, gemaps, egemaps)
            c = custom_180_features(wav_path)
        except Exception as ex:
            print("skip", wav_path, ex)
            continue
        X62.append(g)
        X88.append(e)
        X180.append(c)
        rows.append({"client_id": row["client_id"], "age": row["age"]})

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(args.out_dir, "meta.csv"), index=False)
    np.save(os.path.join(args.out_dir, "X62_gemaps.npy"), np.array(X62))
    np.save(os.path.join(args.out_dir, "X88_egemaps.npy"), np.array(X88))
    np.save(os.path.join(args.out_dir, "X180_custom.npy"), np.array(X180))
    print(f"Saved {len(meta)} clips -> {args.out_dir}/  "
          f"(shapes: {np.array(X62).shape}, {np.array(X88).shape}, {np.array(X180).shape})")


if __name__ == "__main__":
    main()
