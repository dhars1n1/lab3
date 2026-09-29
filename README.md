# 22AIE450 Speech Processing – Lab Assignment 3

## One-command run

Given your dataset layout (`.../cv-corpus-26.0-2026-06-12/hi/clips/*.mp3` +
`validated.tsv` etc.), run everything with:

```bash
cd lab3
python run_all.py --cv_root "/Users/dharsh/Downloads/lab3/cv-corpus-26.0-2026-06-12/hi"
```

That single command:
1. installs every dependency (`pip install -r requirements.txt`),
2. runs Q1's LP-residual/formant analysis (on a dataset clip as a stand-in —
   see note below),
3. extracts all three Q2 feature sets, trains/compares every ML+DL model,
   renders mel-spectrogram images and trains the 2D CNN,
4. fine-tunes both Wav2Vec2-XLS-R-53 and HuBERT with CTC loss and reports
   WER/CER,
5. and drops **everything you need for the report** into `lab3_outputs/`.

It runs in **quick mode** by default — small clip subsets and few epochs —
so the whole thing finishes in a reasonable time on a laptop and you get a
complete, working set of results/plots to build the report structure around.
Once you've checked it end-to-end, rerun with `--full` for the real numbers:

```bash
python run_all.py --cv_root ".../hi" --full
```

**Q1 needs your own voice.** The assignment requires *you* to record
"My name is `<your name>`". Record it first with
`python part1_lp_residual/record_utterance.py`, then pass it in:

```bash
python run_all.py --cv_root ".../hi" --utterance my_name_is_utterance.wav
```

If you don't pass `--utterance`, `run_all.py` automatically substitutes a
random clip from the dataset just so the pipeline completes end-to-end, and
writes a note (`Q1_LP_residual/NOTE_replace_with_own_recording.txt`)
reminding you to redo that step with your real recording before submitting.

Other useful flags:
```bash
--skip_asr        # skip Q3 fine-tuning (by far the slowest stage; needs a GPU for --full)
--skip_install    # skip the automatic pip install (if your env is already set up)
```

### Output layout (`lab3_outputs/`)
```
lab3_outputs/
  Q1_LP_residual/
    lp_analysis.png                 frame + residual + LP spectrum w/ formants
    NOTE_replace_with_own_recording.txt   (only if --utterance wasn't given)
  Q2_age_classification/
    features/                       X62_gemaps.npy, X88_egemaps.npy, X180_custom.npy, meta.csv
    feature_comparison_results.csv  accuracy / macro-F1 for every model x feature-set
    feature_comparison.png
    mel_images/                     bare mel-spectrogram PNGs (no axes/colorbar) + image_meta.csv
    cnn_training_curve.png
    cnn_results.txt
  Q3_asr_finetune/
    wav2vec2/final_model/, metrics.json   (WER/CER)
    hubert/final_model/,    metrics.json  (WER/CER)
  RUN_SUMMARY.txt                   OK/FAILED per stage
  _logs/                            full console log per stage, for debugging
```

Everything above — plots, CSVs, metrics, trained model checkpoints — is what
you attach/quote in the PDF report the assignment asks for.

## What changed in the latest version
- Q1: the voiced frame is now chosen by pitch-periodicity (after DC removal + pre-emphasis), so it is a genuinely voiced frame. Any `.wav` placed directly in the `lab3/` folder is picked up automatically as your utterance (any sample rate / stereo is fine, it is resampled to 16 kHz mono internally).
- Q2: quick mode samples up to 100 clips **per age class** (spread across speakers) instead of random clips, and the speaker-independent split guarantees every age class appears in both train and test. Classical models use balanced class weights.
- Q3: uses the real `facebook/wav2vec2-large-xlsr-53` checkpoint, loads audio with librosa (no torchaudio/torchcodec), runs on CPU when no CUDA GPU is present (MPS lacks CTC-loss support), and also writes `sample_transcripts.txt`.

## Known macOS gotchas

- **XGBoost fails to import** (`libomp.dylib` not loaded): the script now
  catches this and just skips XGBoost, but if you want its numbers, run
  `brew install libomp` once.
- **TensorFlow fails to import on Apple Silicon (M1/M2/M3)**: plain
  `pip install tensorflow` doesn't work there. Install with:
  `pip install tensorflow-macos tensorflow-metal`. The script now skips the
  CNN/MLP stages with a clear message instead of crashing if this isn't set up.
- **`torchaudio` fails to load** (symbol/version mismatch between `torch` and
  `torchaudio`): Part 3 no longer imports `torchaudio` at all — it isn't
  needed, `datasets` + `soundfile` handle the Common Voice `.mp3` decoding on
  their own.

---

## Folder guide (what each script does, if you want to run stages individually)

```
part1_lp_residual/
    record_utterance.py       Q1(a) record "My name is <name>" @16kHz/16-bit
    lp_residual_formants.py   Q1(b,c,d) LP residual + formant estimation
    demo_lp_analysis.png      demo run on a sample voiced frame (included)

part2_age_classification/
    extract_features.py       Q2 extract 62 / 88 / 180-dim feature sets
    train_compare_features.py Q2(i) speaker-independent ML+DL comparison
    melspec_images_cnn.py     Q2(ii) mel-spectrogram images + 2D CNN

part3_asr_finetune/
    finetune_ctc.py           Q3 fine-tune HuBERT / Wav2Vec2-XLS-R-53 with CTC,
                               reports WER & CER

requirements.txt
```

## Run order

### Q1 – Recording & LP analysis
```bash
pip install sounddevice soundfile librosa numpy scipy matplotlib
python part1_lp_residual/record_utterance.py            # run on your own machine (mic)
python part1_lp_residual/lp_residual_formants.py my_name_is_utterance.wav
```
This saves `lp_analysis.png` (frame, residual, LP spectrum with F1–F4 marked)
and prints the formant frequencies / bandwidths and the residual-to-signal
energy ratio, which you can quote directly in your inference.

**What to write in the inference (guidance, not filler for you to skip):**
- The residual is small and impulse-like at glottal closure instants and
  near-white/flat in between — this is because the LP filter models the
  vocal-tract (formant) resonances, so what's *left over* is essentially the
  glottal excitation source (pitch pulses), confirming the source-filter
  model of speech production.
- Formants are read off as the peaks of the LP spectrum / roots of the LPC
  polynomial close to the unit circle (small bandwidth).

### Q2 – Age classification on Common Voice Hindi
1. Download & unzip the dataset from the link in the assignment; note the
   `hi/` folder path (contains `clips/` and `validated.tsv`).
2. Extract all 3 feature sets:
   ```bash
   pip install opensmile librosa pandas tqdm
   python part2_age_classification/extract_features.py \
          --cv_root /path/to/cv-corpus-26.0-2026-06-12/hi --out_dir features/
   ```
3. Compare features + models (speaker-independent split):
   ```bash
   pip install scikit-learn xgboost tensorflow
   python part2_age_classification/train_compare_features.py --features_dir features/
   ```
   Produces `feature_comparison_results.csv` and `feature_comparison.png`.
4. 2D CNN on the raw mel-spectrogram image:
   ```bash
   python part2_age_classification/melspec_images_cnn.py --stage images \
          --cv_root /path/to/hi --out_dir mel_images
   python part2_age_classification/melspec_images_cnn.py --stage train \
          --img_dir mel_images
   ```

Tip: start with `--max_clips 500` while debugging, then rerun on the full
validated split once the pipeline works end-to-end.

### Q3 – Fine-tuning HuBERT / Wav2Vec2-XLS-R-53
```bash
pip install transformers datasets accelerate torch torchaudio jiwer
python part3_asr_finetune/finetune_ctc.py --model wav2vec2 \
       --cv_root /path/to/hi --out_dir runs/w2v2 --max_clips 2000
python part3_asr_finetune/finetune_ctc.py --model hubert \
       --cv_root /path/to/hi --out_dir runs/hubert --max_clips 2000
```
Each run prints and saves final WER/CER in `runs/<model>/metrics.json`.
A GPU (Colab / Kaggle) is strongly recommended — full fine-tuning on CPU is
impractically slow.

## Notes
- All splits use `GroupShuffleSplit`/`GroupKFold` keyed on `client_id` so the
  same speaker never appears in both train and test — this is the
  "speaker-independent" requirement across every experiment.
- `xgboost` / `tensorflow` are optional — the scripts degrade gracefully
  (skip that model) if not installed.
