#!/usr/bin/env python3
"""
run_all.py - run the ENTIRE Speech Processing Lab 3 assignment with one command.

Given the Common Voice Hindi folder layout you have:

    cv-corpus-26.0-2026-06-12/
        hi/
            clips/*.mp3
            validated.tsv, train.tsv, test.tsv, dev.tsv, ...

Just run:

    python run_all.py --cv_root /Users/dharsh/Downloads/lab3/cv-corpus-26.0-2026-06-12/hi

Everything the assignment needs (Q1 LP-residual/formant plot, Q2 feature
comparison + 2D CNN, Q3 HuBERT/Wav2Vec2-XLS-R fine-tuning) is produced inside
one output folder (default: ./lab3_outputs/), ready to drop into your report.

Useful flags:
    --utterance path/to/my_name_is.wav   use YOUR OWN recording for Q1
                                          (otherwise a dataset clip is used as
                                          a placeholder and you'll be warned)
    --quick        (default) small subsets / few epochs so the whole pipeline
                   finishes in a reasonable time on a laptop, to prove
                   everything works end-to-end
    --full         use the full validated split and full epoch counts
                   (slow; a GPU is strongly recommended for Q3)
    --skip_asr     skip Q3 (HuBERT / Wav2Vec2 fine-tuning) - it is by far the
                   slowest stage and needs a GPU for a full run
    --skip_install do not auto-run `pip install -r requirements.txt` first
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, log_path, description):
    """Run a subprocess, tee output to a log file, return True on success."""
    print(f"\n{'='*70}\n>>> {description}\n$ {' '.join(cmd)}\n{'='*70}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.time()
    env = dict(os.environ, PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3",
               PYTORCH_ENABLE_MPS_FALLBACK="1", TOKENIZERS_PARALLELISM="false")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env)
        log_f.write(proc.stdout or "")
    # stream last part of the log to console so the user sees progress/errors
    tail = (proc.stdout or "").strip().splitlines()[-25:]
    print("\n".join(tail))
    elapsed = time.time() - start
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"<<< {description}: {status}  ({elapsed:.1f}s)  [full log: {log_path}]")
    return ok


def pick_demo_clip(cv_root, tsv_name="validated.tsv"):
    import pandas as pd
    df = pd.read_csv(os.path.join(cv_root, tsv_name), sep="\t")
    row = df.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
    return os.path.join(cv_root, "clips", row["path"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv_root", required=True,
                     help="path to the 'hi' folder (contains clips/ and validated.tsv)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "lab3_outputs"))
    ap.add_argument("--utterance", default=None,
                     help="your own 'My name is ...' recording for Q1 (16kHz/16-bit wav)")
    ap.add_argument("--quick", dest="quick", action="store_true", default=True)
    ap.add_argument("--full", dest="quick", action="store_false")
    ap.add_argument("--skip_asr", action="store_true")
    ap.add_argument("--skip_install", action="store_true")
    ap.add_argument("--tsv_name", default="validated.tsv")
    args = ap.parse_args()

    cv_root = os.path.abspath(args.cv_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "_logs")

    if not os.path.isdir(os.path.join(cv_root, "clips")):
        sys.exit(f"ERROR: {cv_root}/clips not found. Point --cv_root at the 'hi' folder "
                  f"(the one that directly contains 'clips/' and 'validated.tsv').")

    os.makedirs(logs_dir, exist_ok=True)
    if not args.skip_install:
        req = os.path.join(HERE, "requirements.txt")
        print(f"Installing requirements from {req} (use --skip_install to skip this) ...")
        run([PY, "-m", "pip", "install", "-q", "-r", req],
            os.path.join(logs_dir, "pip_install.log"), "Installing dependencies")

    # ---- quick vs full scale settings ----
    if args.quick:
        cap_per_class = 100        # balanced: <=100 clips per age class
        cnn_epochs = 15
        mlp_epochs = 40
        asr_max_clips = 40
        asr_epochs = 3
        asr_batch = 2
        print("Running in --quick mode (small subsets, few epochs) so the full "
              "pipeline finishes on a laptop. Re-run with --full for the real "
              "numbers you'd report.")
    else:
        cap_per_class = None       # all validated clips with an age label
        cnn_epochs = 25
        mlp_epochs = 60
        asr_max_clips = None
        asr_epochs = 10
        asr_batch = 4

    results = {}

    # ================= Q1: LP residual + formants ================= #
    q1_dir = os.path.join(out_dir, "Q1_LP_residual")
    os.makedirs(q1_dir, exist_ok=True)

    utterance = args.utterance
    if utterance is None:
        import glob
        own = sorted(glob.glob(os.path.join(HERE, "*.wav")))
        if own:
            utterance = own[0]
            print(f"[Q1] using the recording found in the lab3 folder: {utterance}")
    if utterance is None:
        utterance = pick_demo_clip(cv_root, args.tsv_name)
        print(f"\n[Q1] NOTE: --utterance not given, using a DATASET clip as a "
              f"stand-in so the pipeline runs end-to-end:\n      {utterance}\n"
              f"      Replace this with your own recording of "
              f'"My name is <your name>" (see part1_lp_residual/record_utterance.py) '
              f"and re-run with --utterance for the real submission.")
        with open(os.path.join(q1_dir, "NOTE_replace_with_own_recording.txt"), "w") as f:
            f.write("This run used a Common Voice clip as a placeholder for Q1:\n"
                    f"{utterance}\n\n"
                    "Record your own utterance with record_utterance.py and re-run:\n"
                    "  python run_all.py --cv_root <...> --utterance my_name_is.wav\n")

    ok = run(
        [PY, os.path.join(HERE, "part1_lp_residual", "lp_residual_formants.py"),
         utterance, "--out_prefix", os.path.join(q1_dir, "lp_analysis")],
        os.path.join(logs_dir, "q1_lp_analysis.log"),
        "Q1: LP residual + formant estimation",
    )
    results["Q1_lp_residual"] = ok

    # ================= Q2: feature extraction ================= #
    q2_dir = os.path.join(out_dir, "Q2_age_classification")
    features_dir = os.path.join(q2_dir, "features")
    cmd = [PY, os.path.join(HERE, "part2_age_classification", "extract_features.py"),
           "--cv_root", cv_root, "--out_dir", features_dir]
    if cap_per_class:
        cmd += ["--cap_per_class", str(cap_per_class)]
    ok = run(cmd, os.path.join(logs_dir, "q2_extract_features.log"),
              "Q2: extracting 62 / 88 / 180-dim feature sets")
    results["Q2_extract_features"] = ok

    # ================= Q2(i): ML+DL comparison ================= #
    if ok:
        cmd = [PY, os.path.join(HERE, "part2_age_classification", "train_compare_features.py"),
               "--features_dir", features_dir, "--out_dir", q2_dir, "--epochs", str(mlp_epochs)]
        ok_cmp = run(cmd, os.path.join(logs_dir, "q2_train_compare.log"),
                      "Q2(i): speaker-independent ML/DL comparison across feature sets")
        results["Q2_feature_comparison"] = ok_cmp
    else:
        print("[Q2(i)] skipped - feature extraction failed")
        results["Q2_feature_comparison"] = False

    # ================= Q2(ii): mel-spectrogram images + 2D CNN ================= #
    mel_dir = os.path.join(q2_dir, "mel_images")
    cmd = [PY, os.path.join(HERE, "part2_age_classification", "melspec_images_cnn.py"),
           "--stage", "images", "--cv_root", cv_root, "--tsv_name", args.tsv_name,
           "--out_dir", mel_dir]
    if cap_per_class:
        cmd += ["--cap_per_class", str(cap_per_class)]
    ok_img = run(cmd, os.path.join(logs_dir, "q2_melspec_images.log"),
                  "Q2(ii): rendering mel-spectrogram images")
    results["Q2_melspec_images"] = ok_img

    if ok_img:
        cmd = [PY, os.path.join(HERE, "part2_age_classification", "melspec_images_cnn.py"),
               "--stage", "train", "--img_dir", mel_dir, "--epochs", str(cnn_epochs),
               "--results_dir", q2_dir]
        ok_cnn = run(cmd, os.path.join(logs_dir, "q2_cnn_train.log"),
                      "Q2(ii): training the 2D CNN on mel-spectrogram images")
        results["Q2_cnn"] = ok_cnn
    else:
        print("[Q2(ii)] skipped CNN training - image rendering failed")
        results["Q2_cnn"] = False

    # ================= Q3: ASR fine-tuning ================= #
    if args.skip_asr:
        print("\n[Q3] skipped (--skip_asr)")
        results["Q3_wav2vec2"] = None
        results["Q3_hubert"] = None
    else:
        q3_dir = os.path.join(out_dir, "Q3_asr_finetune")
        for model_key in ["wav2vec2", "hubert"]:
            run_dir = os.path.join(q3_dir, model_key)
            cmd = [PY, os.path.join(HERE, "part3_asr_finetune", "finetune_ctc.py"),
                   "--model", model_key, "--cv_root", cv_root, "--tsv_name", args.tsv_name,
                   "--out_dir", run_dir, "--epochs", str(asr_epochs),
                   "--batch_size", str(asr_batch)]
            if asr_max_clips:
                cmd += ["--max_clips", str(asr_max_clips)]
            ok_asr = run(cmd, os.path.join(logs_dir, f"q3_{model_key}.log"),
                          f"Q3: fine-tuning {model_key} with CTC loss")
            results[f"Q3_{model_key}"] = ok_asr

    # ================= summary ================= #
    summary_path = os.path.join(out_dir, "RUN_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("Lab 3 - run_all.py summary\n" + "=" * 40 + "\n")
        f.write(f"cv_root: {cv_root}\nmode: {'quick' if args.quick else 'full'}\n\n")
        for k, v in results.items():
            status = "SKIPPED" if v is None else ("OK" if v else "FAILED")
            f.write(f"{k:30s} {status}\n")
        f.write(f"\nAll outputs are under: {out_dir}\n"
                "See _logs/ for the full console output of every stage.\n")

    print("\n" + "=" * 70)
    print(open(summary_path).read())
    print(f"Everything is in: {out_dir}")


if __name__ == "__main__":
    main()
