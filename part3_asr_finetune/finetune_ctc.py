"""
Q3 - Fine-tune HuBERT and Wav2Vec2-XLS-R-53 with CTC loss for Hindi ASR.
Evaluates with WER and CER (jiwer).

    facebook/wav2vec2-large-xlsr-53   (XLS-R-53 family)
    facebook/hubert-large-ll60k

Audio is loaded with librosa (no dependency on torchaudio / torchcodec).

    python finetune_ctc.py --model wav2vec2 --cv_root /path/to/hi --out_dir runs/w2v2
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import librosa
import torch
import jiwer
from datasets import Dataset
from transformers import (
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

SR = 16000
MODEL_CKPT = {
    "wav2vec2": "facebook/wav2vec2-large-xlsr-53",
    "hubert": "facebook/hubert-large-ll60k",
}
MODEL_CLASS = {"wav2vec2": Wav2Vec2ForCTC, "hubert": HubertForCTC}
IGNORE_RE = re.compile(r"[,?.!\-;:\"“%‘”�।'’()\[\]|]")
MAX_SECONDS = 10.0


def load_df(cv_root, tsv_name):
    df = pd.read_csv(os.path.join(cv_root, tsv_name), sep="\t", quoting=3,
                     usecols=["client_id", "path", "sentence"])
    df["path"] = df["path"].apply(lambda p: os.path.join(cv_root, "clips", p))
    df["sentence"] = df["sentence"].astype(str).apply(lambda s: IGNORE_RE.sub("", s).strip())
    df = df[df["sentence"].str.len() > 0]
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


def speaker_split(df, test_frac=0.2, seed=42):
    spk = df["client_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(spk)
    n_test = max(1, int(round(len(spk) * test_frac)))
    if len(spk) < 2:                       # degenerate: fall back to clip split
        n = max(1, len(df) // 5)
        return df.iloc[n:].reset_index(drop=True), df.iloc[:n].reset_index(drop=True)
    test_spk = set(spk[:n_test])
    te = df[df["client_id"].isin(test_spk)].reset_index(drop=True)
    tr = df[~df["client_id"].isin(test_spk)].reset_index(drop=True)
    return tr, te


def build_vocab(sentences, path):
    chars = sorted(set("".join(sentences)) - {" "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab


@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor

    def __call__(self, feats: List[Dict[str, Union[List[int], List[float]]]]):
        pad = self.processor.feature_extractor.pad(
            [{"input_values": f["input_values"]} for f in feats],
            padding=True, return_tensors="pt", return_attention_mask=True)
        tok = self.processor.tokenizer
        max_len = max(len(f["labels"]) for f in feats)
        labels = torch.full((len(feats), max_len), -100, dtype=torch.long)
        for i, f in enumerate(feats):
            labels[i, :len(f["labels"])] = torch.tensor(f["labels"])
        batch = {"input_values": pad["input_values"], "labels": labels}
        if "attention_mask" in pad:
            batch["attention_mask"] = pad["attention_mask"]
        return batch


def make_metrics(processor):
    tok = processor.tokenizer

    def compute(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        hyp = tok.batch_decode(ids)
        ref = tok.batch_decode(label_ids, group_tokens=False)
        # jiwer can't score empty references
        pairs = [(r, h) for r, h in zip(ref, hyp) if r.strip()]
        ref, hyp = [p[0] for p in pairs], [p[1] for p in pairs]
        return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CKPT), required=True)
    ap.add_argument("--cv_root", required=True)
    ap.add_argument("--tsv_name", default="validated.tsv")
    ap.add_argument("--out_dir", default="run")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_df(args.cv_root, args.tsv_name)
    if args.max_clips:
        df = df.sample(n=min(args.max_clips, len(df)), random_state=42).reset_index(drop=True)
    train_df, test_df = speaker_split(df)
    print(f"train clips: {len(train_df)}  test clips: {len(test_df)}  "
          f"(speaker-independent)")

    vocab_path = os.path.join(args.out_dir, "vocab.json")
    build_vocab(pd.concat([train_df, test_df])["sentence"].tolist(), vocab_path)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)

    def encode(row):
        y, _ = librosa.load(row["path"], sr=SR, mono=True)
        y = y[: int(MAX_SECONDS * SR)]
        vals = fe(y, sampling_rate=SR).input_values[0]
        text = row["sentence"].replace(" ", "|")
        labels = tokenizer(text).input_ids
        return {"input_values": vals, "labels": labels, "n": len(vals)}

    def to_ds(d):
        rows = []
        for _, r in d.iterrows():
            try:
                rows.append(encode(r))
            except Exception as e:
                print("skip", r["path"], e)
        return Dataset.from_list(rows)

    train_ds, test_ds = to_ds(train_df), to_ds(test_df)

    model = MODEL_CLASS[args.model].from_pretrained(
        MODEL_CKPT[args.model],
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        mask_time_prob=0.05,
        layerdrop=0.0,
    )
    model.freeze_feature_encoder()
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    ta = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    kw["eval_strategy" if "eval_strategy" in ta else "evaluation_strategy"] = "no"
    if not torch.cuda.is_available():
        # MPS has no CTC-loss kernel -> force plain CPU (reliable). Use a GPU for --full.
        if "use_cpu" in ta:
            kw["use_cpu"] = True
        elif "no_cuda" in ta:
            kw["no_cuda"] = True

    collator = CTCCollator(processor)
    tr_kwargs = dict(model=model, args=TrainingArguments(**kw), train_dataset=train_ds,
                     eval_dataset=test_ds, data_collator=collator,
                     compute_metrics=make_metrics(processor))
    tr_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in tr_params:
        tr_kwargs["processing_class"] = processor.feature_extractor
    elif "tokenizer" in tr_params:
        tr_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Trainer(**tr_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    wer, cer = metrics.get("eval_wer"), metrics.get("eval_cer")
    print(f"\n=== {args.model} ({MODEL_CKPT[args.model]}) ===\nWER: {wer:.4f}   CER: {cer:.4f}")

    # a few example transcripts for the report
    preds = trainer.predict(test_ds.select(range(min(8, len(test_ds)))))
    ids = np.argmax(preds.predictions, axis=-1)
    lab = np.where(preds.label_ids == -100, tokenizer.pad_token_id, preds.label_ids)
    with open(os.path.join(args.out_dir, "sample_transcripts.txt"), "w", encoding="utf-8") as f:
        for r, h in zip(tokenizer.batch_decode(lab, group_tokens=False), tokenizer.batch_decode(ids)):
            f.write(f"REF: {r}\nHYP: {h}\n\n")

    trainer.save_model(os.path.join(args.out_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.out_dir, "final_model"))
    json.dump({"model": MODEL_CKPT[args.model], "wer": wer, "cer": cer,
               "train_clips": len(train_ds), "test_clips": len(test_ds)},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
