"""
Custom ASR Model Training Script
Fine-tunes OpenAI Whisper on your own audio dataset for a single target language.

Supports two data formats (auto-detected):
  - transcripts.txt : pipe-separated  →  filename|transcript text
  - metadata.json   : JSON list of {filename, text, ...}

Usage:
  python train_asr_model.py                        # train with defaults
  python train_asr_model.py --language ceb         # Cebuano
  python train_asr_model.py --model_size small --epochs 10
  python train_asr_model.py --mode infer --audio_path path/to/test.wav
"""

import os

import re
import json
import argparse
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch

import numpy as np
import librosa
import evaluate
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
ffmpeg_bin = r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build-shared\bin"
if os.path.exists(ffmpeg_bin):
    os.add_dll_directory(ffmpeg_bin)
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (override via CLI or edit defaults here)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Paths
    "audio_dir":        "audio_data",
    "transcript_file":  "audio_data/transcripts.txt",   # pipe-separated
    "metadata_file":    "audio_data/metadata.json",     # JSON alternative
    "output_dir":       "asr_model_output",
    "model_save_path":  "asr_model_final",

    # Model
    "base_model":       "openai/whisper-small",  # tiny | base | small | medium | large-v3
    "language":         "en",                    # BCP-47 code  e.g. "ceb", "tl", "en"
    "task":             "transcribe",

    # Training
    "test_split_ratio": 0.15,
    "max_input_length_s": 30.0,   # discard audio longer than this (seconds)
    "batch_size":       8,
    "grad_accum_steps": 2,
    "learning_rate":    1e-5,
    "warmup_steps":     100,
    "max_steps":        2000,
    "eval_steps":       200,
    "save_steps":       200,
    "fp16":             True,     # set False on CPU / MPS
    "generation_max_length": 225,
    "num_beams":        5,

    # Inference
    "infer_audio_path": None,
}

WHISPER_SR = 16_000   # Whisper always expects 16 kHz


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """Lowercase and strip punctuation for cleaner WER evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_manifest(audio_dir: str,
                  transcript_file: str,
                  metadata_file: str) -> List[Dict]:
    """
    Returns a list of {"path": str, "sentence": str} dictionaries.
    Auto-detects the format: tries transcripts.txt first, then metadata.json.
    """
    records = []

    if Path(transcript_file).is_file():
        print(f"[Data] Loading from pipe-separated file: {transcript_file}")
        with open(transcript_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = line.split("|", 1)
                stem, text = parts[0].strip(), parts[1].strip()
                # Try both plain stem and stem+.wav
                for candidate in [stem, stem + ".wav"]:
                    full = Path(audio_dir) / candidate
                    if full.is_file():
                        records.append({"path": str(full), "sentence": text})
                        break
                else:
                    print(f"  [WARN] Audio not found for stem '{stem}' – skipping.")

    elif Path(metadata_file).is_file():
        print(f"[Data] Loading from JSON file: {metadata_file}")
        with open(metadata_file, encoding="utf-8") as f:
            meta = json.load(f)
        entries = meta if isinstance(meta, list) else meta.get("recordings", [])
        for entry in entries:
            fname = entry.get("filename", "")
            text  = entry.get("text",     "")
            full  = Path(audio_dir) / fname
            if full.is_file():
                records.append({"path": str(full), "sentence": text})
            else:
                print(f"  [WARN] Audio not found: {full} – skipping.")

    else:
        raise FileNotFoundError(
            f"No data manifest found.\n"
            f"  Expected: {transcript_file}  or  {metadata_file}\n"
            "  Please check your paths in DEFAULT_CONFIG."
        )

    print(f"[Data] Loaded {len(records)} samples.")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(records: List[Dict],
                  test_split: float,
                  config: Dict) -> DatasetDict:
    """Convert records list → HuggingFace DatasetDict with train/test splits."""
    ds = Dataset.from_list(records).cast_column("path", Audio(sampling_rate=WHISPER_SR))

    split = ds.train_test_split(test_size=test_split, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def make_prepare_fn(processor: WhisperProcessor, config: Dict):
    """Returns a map() function that converts raw audio + text to model inputs."""
    max_len = int(config["max_input_length_s"] * WHISPER_SR)

    def prepare(batch):
        audio = batch["path"]   # already loaded & resampled by datasets Audio()
        array = audio["array"]
        sr    = audio["sampling_rate"]

        # Safety: re-resample if needed
        if sr != WHISPER_SR:
            array = librosa.resample(array, orig_sr=sr, target_sr=WHISPER_SR)

        # Drop samples that exceed max length
        if len(array) > max_len:
            return {"input_features": None, "labels": None}

        features = processor.feature_extractor(
            array,
            sampling_rate=WHISPER_SR,
            return_tensors="np",
        ).input_features[0]

        labels = processor.tokenizer(
            batch["sentence"],
            return_tensors="np",
        ).input_ids[0]

        return {"input_features": features, "labels": labels}

    return prepare


# ──────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR  (pads labels to same length in a batch)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Input features are fixed-size mel spectrograms – just stack them
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Labels need padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip decoder BOS if tokenizer prepends it
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ──────────────────────────────────────────────────────────────────────────────
# METRIC
# ──────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 back to pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalise before scoring
        pred_str  = [_normalise_text(s) for s in pred_str]
        label_str = [_normalise_text(s) for s in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(100 * wer, 2)}

    return compute_metrics


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train(config: Dict):
    print("\n=== CUSTOM ASR TRAINING  (Whisper fine-tune) ===\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    records = load_manifest(
        config["audio_dir"],
        config["transcript_file"],
        config["metadata_file"],
    )
    if len(records) == 0:
        raise RuntimeError("No labelled audio samples found. Check your audio_dir and transcript paths.")

    # ── 2. Build HuggingFace processor ───────────────────────────────────────
    print(f"\n[Model] Base: {config['base_model']}   Language: {config['language']}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config["base_model"])
    tokenizer = WhisperTokenizer.from_pretrained(
        config["base_model"],
        language=config["language"],
        task=config["task"],
    )
    processor = WhisperProcessor.from_pretrained(
        config["base_model"],
        language=config["language"],
        task=config["task"],
    )

    # ── 3. Pre-process dataset ─────────────────────────────────────────────
    print("\n[Data] Pre-processing audio…")
    dataset = build_dataset(records, config["test_split_ratio"], config)
    prepare_fn = make_prepare_fn(processor, config)
    dataset = dataset.map(
        prepare_fn,
        remove_columns=["path", "sentence"],
        num_proc=None,       # force single-process map (avoids Python 3.14 multiprocess/dill crash)
    )
    # Drop rows where audio exceeded max length (prepare returned None)
    dataset = dataset.filter(lambda x: x["input_features"] is not None)

    print(f"  Train samples : {len(dataset['train'])}")
    print(f"  Test  samples : {len(dataset['test'])}")

    # ── 4. Load model ─────────────────────────────────────────────────────
    model = WhisperForConditionalGeneration.from_pretrained(config["base_model"])
    model.generation_config.language = config["language"]
    model.generation_config.task     = config["task"]
    model.generation_config.forced_decoder_ids = None   # let generate() handle it

    # ── 5. Data collator & metrics ────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = make_compute_metrics(processor)

    # ── 6. Training arguments ─────────────────────────────────────────────
    use_fp16 = config["fp16"] and torch.cuda.is_available()
    use_bf16 = (not use_fp16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args_kwargs = {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["grad_accum_steps"],
        "learning_rate": config["learning_rate"],
        "warmup_steps": config["warmup_steps"],
        "max_steps": config["max_steps"],
        "gradient_checkpointing": True,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "per_device_eval_batch_size": max(config["batch_size"] // 2, 1),
        "predict_with_generate": True,
        "generation_max_length": config["generation_max_length"],
        "save_steps": config["save_steps"],
        "eval_steps": config["eval_steps"],
        "logging_steps": 25,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "push_to_hub": False,
    }
    if "eval_strategy" in Seq2SeqTrainingArguments.__init__.__code__.co_varnames:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # ── 7. Trainer ────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── 8. Train ──────────────────────────────────────────────────────────
    print("\n[Train] Starting training …")
    trainer.train()

    # ── 9. Final evaluation ───────────────────────────────────────────────
    print("\n[Eval] Running final evaluation …")
    metrics = trainer.evaluate()
    print(f"  Final WER : {metrics.get('eval_wer', 'N/A')} %")

    # ── 10. Save model & processor ────────────────────────────────────────
    save_path = config["model_save_path"]
    print(f"\n[Save] Saving model → {save_path}")
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save config snapshot alongside the model
    config_out = Path(save_path) / "training_config.json"
    with open(config_out, "w") as f:
        serialisable = {k: v for k, v in config.items() if v is not None}
        json.dump(serialisable, f, indent=2)

    print(f"\n✓ Training complete! Model saved to: {save_path}\n")
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def load_model_for_inference(model_path: str):
    """Load a saved model + processor from disk."""
    print(f"[Infer] Loading model from: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def transcribe(audio_path: str,
               model_path: str,
               language: str = "en",
               num_beams: int = 5) -> str:
    """
    Transcribe a single audio file with a fine-tuned Whisper model.

    Args:
        audio_path  : Path to .wav / .mp3 / .flac file.
        model_path  : Directory of the saved fine-tuned model.
        language    : BCP-47 language code used during training.
        num_beams   : Beam width for decoding (higher = slower but better).

    Returns:
        Transcription string.
    """
    model, processor, device = load_model_for_inference(model_path)

    # Load & resample audio
    array, sr = librosa.load(audio_path, sr=WHISPER_SR, mono=True)
    print(f"[Infer] Audio: {audio_path}  |  duration: {len(array)/WHISPER_SR:.2f}s")

    # Extract features
    inputs = processor.feature_extractor(
        array,
        sampling_rate=WHISPER_SR,
        return_tensors="pt",
    ).input_features.to(device)

    # Force the model to decode in the target language
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=num_beams,
        )

    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    return transcription


def batch_transcribe(audio_paths: List[str],
                     model_path: str,
                     language: str = "en",
                     num_beams: int = 5) -> List[Dict]:
    """
    Transcribe a list of audio files.

    Returns a list of {"file": str, "transcription": str} dicts.
    """
    model, processor, device = load_model_for_inference(model_path)
    results = []

    for path in audio_paths:
        try:
            array, _ = librosa.load(path, sr=WHISPER_SR, mono=True)
            inputs = processor.feature_extractor(
                array,
                sampling_rate=WHISPER_SR,
                return_tensors="pt",
            ).input_features.to(device)

            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

            with torch.no_grad():
                pred_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=num_beams,
                )

            text = processor.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )[0].strip()

            results.append({"file": path, "transcription": text})
            print(f"  [{path}] → {text}")

        except Exception as e:
            results.append({"file": path, "transcription": f"ERROR: {e}"})
            print(f"  [{path}] ERROR: {e}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train or run inference with a custom Whisper ASR model.")
    p.add_argument("--mode",           choices=["train", "infer", "batch_infer"],
                   default="train",    help="Operation mode (default: train)")

    # Data / paths
    p.add_argument("--audio_dir",      default=DEFAULT_CONFIG["audio_dir"])
    p.add_argument("--transcript_file",default=DEFAULT_CONFIG["transcript_file"])
    p.add_argument("--metadata_file",  default=DEFAULT_CONFIG["metadata_file"])
    p.add_argument("--output_dir",     default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--model_save_path",default=DEFAULT_CONFIG["model_save_path"])

    # Model
    p.add_argument("--model_size",     choices=["tiny","base","small","medium","large-v3"],
                   default="small",    help="Whisper model size (default: small)")
    p.add_argument("--language",       default=DEFAULT_CONFIG["language"],
                   help="BCP-47 language code, e.g. 'en', 'ceb', 'tl' (default: en)")

    # Training hyper-params
    p.add_argument("--epochs",         type=int, default=None,
                   help="Alternative to max_steps: approximate number of epochs")
    p.add_argument("--max_steps",      type=int, default=DEFAULT_CONFIG["max_steps"])
    p.add_argument("--batch_size",     type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--learning_rate",  type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--no_fp16",        action="store_true", help="Disable mixed-precision (fp16)")
    p.add_argument("--test_split",     type=float, default=DEFAULT_CONFIG["test_split_ratio"])

    # Inference
    p.add_argument("--audio_path",     default=None,    help="Single audio file to transcribe")
    p.add_argument("--audio_list",     default=None,    help="Text file with one audio path per line")
    p.add_argument("--num_beams",      type=int,  default=DEFAULT_CONFIG["num_beams"])

    return p.parse_args()


def main():
    args = _parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "audio_dir":         args.audio_dir,
        "transcript_file":   args.transcript_file,
        "metadata_file":     args.metadata_file,
        "output_dir":        args.output_dir,
        "model_save_path":   args.model_save_path,
        "base_model":       f"openai/whisper-{args.model_size}",
        "language":          args.language,
        "max_steps":         args.max_steps,
        "batch_size":        args.batch_size,
        "learning_rate":     args.learning_rate,
        "fp16":             not args.no_fp16,
        "test_split_ratio":  args.test_split,
        "num_beams":         args.num_beams,
    })

    # ── TRAIN ─────────────────────────────────────────────────────────────
    if args.mode == "train":
        train(config)

    # ── SINGLE INFER ──────────────────────────────────────────────────────
    elif args.mode == "infer":
        if not args.audio_path:
            raise ValueError("--audio_path is required for --mode infer")
        result = transcribe(
            audio_path=args.audio_path,
            model_path=config["model_save_path"],
            language=config["language"],
            num_beams=config["num_beams"],
        )
        print(f"\nTranscription:\n  {result}\n")

    # ── BATCH INFER ───────────────────────────────────────────────────────
    elif args.mode == "batch_infer":
        if not args.audio_list:
            raise ValueError("--audio_list is required for --mode batch_infer")
        with open(args.audio_list) as f:
            paths = [line.strip() for line in f if line.strip()]
        results = batch_transcribe(
            audio_paths=paths,
            model_path=config["model_save_path"],
            language=config["language"],
            num_beams=config["num_beams"],
        )
        # Write results to CSV
        out_csv = "batch_transcription_results.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "transcription"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Batch results saved → {out_csv}")


if __name__ == "__main__":
    main()
