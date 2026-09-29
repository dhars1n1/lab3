"""
Q2 (i) - Speaker-independent age classification: compare the 3 feature sets
   - 62  : GeMAPS
   - 88  : eGeMAPS
   - 180 : MFCC(40) + Mel-spectrogram(128) + Chroma(12)

Models compared:
   Classical ML : SVM (RBF), Random Forest, Logistic Regression, XGBoost
   Deep Learning: a small MLP (Keras)

Speaker-independence is enforced with GroupKFold / GroupShuffleSplit keyed
on `client_id`, so no speaker appears in both train and test.

Usage:
    python train_compare_features.py --features_dir features/
"""

import argparse
import os
import numpy as np
import pandas as pd
from split_utils import speaker_independent_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as _xgb_err:
    HAS_XGB = False
    print(f"[warn] XGBoost unavailable ({type(_xgb_err).__name__}: {_xgb_err}) - skipping it.")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except Exception as _tf_err:
    HAS_TF = False
    print(f"[warn] TensorFlow unavailable ({type(_tf_err).__name__}: {_tf_err}) "
          f"- the MLP(DL) model will be skipped for all feature sets.\n"
          f"       On Apple Silicon Macs, install with:\n"
          f"         pip install tensorflow-macos tensorflow-metal\n"
          f"       (plain 'pip install tensorflow' does not work on M1/M2/M3).")


FEATURE_SETS = {
    "62_GeMAPS": "X62_gemaps.npy",
    "88_eGeMAPS": "X88_egemaps.npy",
    "180_Custom": "X180_custom.npy",
}


def build_mlp(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate(y_true, y_pred, label_encoder, model_name, feature_name, results):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    results.append({"feature_set": feature_name, "model": model_name,
                     "accuracy": acc, "macro_f1": f1})
    print(f"[{feature_name} | {model_name}] acc={acc:.4f}  macro-F1={f1:.4f}")
    print(classification_report(y_true, y_pred,
                                 labels=range(len(label_encoder.classes_)),
                                 target_names=label_encoder.classes_,
                                 zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out_dir", default=".",
                     help="where to write feature_comparison_results.csv/png")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(args.features_dir, "meta.csv"))
    le = LabelEncoder()
    y = le.fit_transform(meta["age"])

    idx_train, idx_test = speaker_independent_split(y, meta["client_id"].values)
    results = []

    for feat_name, fname in FEATURE_SETS.items():
        X = np.load(os.path.join(args.features_dir, fname))
        Xtr, Xte = X[idx_train], X[idx_test]
        ytr, yte = y[idx_train], y[idx_test]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        # ---- Classical ML models ----
        try:
            svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced").fit(Xtr_s, ytr)
            evaluate(yte, svm.predict(Xte_s), le, "SVM-RBF", feat_name, results)
        except Exception as e:
            print(f"[skip] SVM-RBF on {feat_name}: {e}")

        try:
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced").fit(Xtr, ytr)
            evaluate(yte, rf.predict(Xte), le, "RandomForest", feat_name, results)
        except Exception as e:
            print(f"[skip] RandomForest on {feat_name}: {e}")

        try:
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr)
            evaluate(yte, lr.predict(Xte_s), le, "LogReg", feat_name, results)
        except Exception as e:
            print(f"[skip] LogReg on {feat_name}: {e}")

        if HAS_XGB:
            try:
                xgb = XGBClassifier(n_estimators=300, max_depth=6,
                                     eval_metric="mlogloss").fit(Xtr, ytr)
                evaluate(yte, xgb.predict(Xte), le, "XGBoost", feat_name, results)
            except Exception as e:
                print(f"[skip] XGBoost on {feat_name}: {e}")

        # ---- Deep learning: MLP ----
        if HAS_TF:
            try:
                mlp = build_mlp(Xtr_s.shape[1], len(le.classes_))
                mlp.fit(Xtr_s, ytr, validation_split=0.1, epochs=args.epochs,
                        batch_size=32, verbose=0)
                y_pred = np.argmax(mlp.predict(Xte_s), axis=1)
                evaluate(yte, y_pred, le, "MLP(DL)", feat_name, results)
            except Exception as e:
                print(f"[skip] MLP(DL) on {feat_name}: {e}")

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No model/feature-set combination succeeded - check the errors above "
              "(e.g. a class with too few speakers for the current --max_clips subset; "
              "try --full or a larger sample).")
        return
    csv_path = os.path.join(args.out_dir, "feature_comparison_results.csv")
    res_df.to_csv(csv_path, index=False)
    print("\n=== Summary ===")
    print(res_df.pivot(index="model", columns="feature_set", values="accuracy"))

    # bar plot for the report
    pivot = res_df.pivot(index="model", columns="feature_set", values="accuracy")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Accuracy")
    plt.title("Age classification accuracy: 62 vs 88 vs 180 features")
    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "feature_comparison.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
