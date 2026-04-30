# Custom ASR Model Training Guide

Fine-tuning OpenAI **Whisper** on your own single-language audio dataset using `train_asr_model.py`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Collection](#3-data-collection)
4. [Data Format & Structure](#4-data-format--structure)
5. [Training the Model](#5-training-the-model)
6. [Monitoring Training](#6-monitoring-training)
7. [Saving & Versioning the Model](#7-saving--versioning-the-model)
8. [Inference](#8-inference)
9. [Evaluating Model Quality](#9-evaluating-model-quality)
10. [Tips & Troubleshooting](#10-tips--troubleshooting)
11. [Language Code Reference](#11-language-code-reference)
12. [Repository Scope, License, and Legal Disclaimers](#12-repository-scope-license-and-legal-disclaimers)

---

## 1. Overview

The script fine-tunes a pre-trained **Whisper** model (any size) using your labelled audio, locking it to a **single target language**. Whisper is a transformer-based encoder-decoder that accepts raw 16 kHz audio and outputs text.

```
Raw audio (.wav) + Transcripts (.txt / .json)
        │
        ▼
  WhisperFeatureExtractor → 80-channel log-mel spectrogram
        │
        ▼
  Whisper encoder → decoder (fine-tuned on your data)
        │
        ▼
  Transcription text
```

**Minimum viable dataset:** ~1 hour of clean, labelled speech.  
**Recommended:** 3–10+ hours for production-quality results.

---

## 2. Environment Setup

### 2.1 Create and activate a virtual environment

```bash
python -m venv asr_env
source asr_env/bin/activate       # macOS / Linux
# asr_env\Scripts\activate        # Windows
```

### 2.2 Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only (no GPU):
# pip install torch torchvision torchaudio

pip install transformers>=4.40.0 \
            datasets>=2.18.0 \
            accelerate>=0.28.0 \
            evaluate \
            jiwer \
            librosa \
            tensorboard \
            soundfile
```

### 2.3 Verify GPU availability

```python
import torch
print(torch.cuda.is_available())          # True = NVIDIA GPU ready
print(torch.backends.mps.is_available())  # True = Apple Silicon GPU ready
```

> **No GPU?** Training still works on CPU but will be very slow (hours per epoch).  
> Use `--no_fp16` flag and reduce `--max_steps` for testing.

---

## 3. Data Collection

### 3.1 What makes good training data

| Property | Requirement |
|----------|-------------|
| Language | **Single language only** (set via `--language`) |
| Audio quality | Clean, minimal background noise and echo |
| Sample rate | Any (script auto-resamples to 16 kHz) |
| Format | `.wav`, `.mp3`, `.flac`, `.ogg` |
| Utterance length | 2–25 seconds per file (ideal: 5–15 s) |
| Transcripts | Accurate, verbatim text for each file |
| Speakers | Diverse voices improve generalisation |

### 3.2 Collection methods

**Option A – Record yourself (quick start)**

Use `audio_data_creator.py` already in this project, or any recording tool (Audacity, Voice Memos, etc.).

```bash
python audio_data_creator.py
```

**Option B – Public datasets** (pre-labelled, ready to use)

| Dataset | Language | Hours | Link |
|---------|----------|-------|------|
| Mozilla Common Voice | 100+ langs | Varies | https://commonvoice.mozilla.org |
| LibriSpeech | English | 960 h | https://openslr.org/12 |
| OpenSLR (Cebuano / Filipino) | Various | Varies | https://openslr.org |
| AISHELL | Mandarin | 170 h | https://openslr.org/33 |
| VoxPopuli | EU langs | Varies | https://github.com/facebookresearch/voxpopuli |

**Option C – Synthetic data augmentation**

If you have limited real data, augment with:
- Speed perturbation (0.9×, 1.1×)  
- Adding light background noise  
- Room impulse response (RIR) convolution  

Use `audiomentations` or `torchaudio.transforms` for augmentation.

### 3.3 Minimum dataset breakdown

```
Total: 1 hour minimum
  ├── 85% → training (~51 min)
  └── 15% → evaluation (~9 min)
```

---

## 4. Data Format & Structure

The script auto-detects one of two formats.

### 4.1 Format A – Pipe-separated transcript file (default)

**File:** `audio_data/transcripts.txt`

```
filename_without_extension|Exact transcript text here
Eric_20250527_212255|The quick brown fox jumps over the lazy dog.
Eric_20250527_212317|She sells seashells by the seashore.
```

- One line per audio file
- Left of `|` = file stem (with or without `.wav`)
- Right of `|` = verbatim transcript

### 4.2 Format B – JSON metadata file

**File:** `audio_data/metadata.json`

```json
{
  "recordings": [
    {
      "filename": "Eric_20250527_212255.wav",
      "text": "The quick brown fox jumps over the lazy dog.",
      "speaker": "Eric",
      "duration": 5.2
    }
  ]
}
```

### 4.3 Folder structure

```
VoiceAndAudio/
├── train_asr_model.py
├── audio_data/
│   ├── transcripts.txt          ← manifest
│   ├── Eric_20250527_212255.wav
│   ├── Eric_20250527_212317.wav
│   └── ...
├── asr_model_output/            ← created during training (checkpoints)
└── asr_model_final/             ← created on completion (final model)
```

---

## 5. Training the Model

### 5.1 Basic training (English, Whisper-small)

```bash
python train_asr_model.py
```

### 5.2 Training for a specific language

```bash
# Cebuano
python train_asr_model.py --language ceb

# Filipino / Tagalog
python train_asr_model.py --language tl

# French
python train_asr_model.py --language fr
```

### 5.3 Choosing a model size

| Size | Parameters | GPU RAM | WER (typical) | Speed |
|------|-----------|---------|---------------|-------|
| `tiny` | 39 M | ~1 GB | Higher | Fastest |
| `base` | 74 M | ~1.5 GB | Good | Fast |
| `small` | 244 M | ~3 GB | Better | Moderate |
| `medium` | 769 M | ~6 GB | Best practical | Slow |
| `large-v3` | 1.5 B | ~12 GB | Highest | Very slow |

```bash
# Use base for low-resource environments
python train_asr_model.py --model_size base --language ceb

# Use medium for best quality (requires GPU)
python train_asr_model.py --model_size medium --language ceb
```

### 5.4 Full example with custom parameters

```bash
python train_asr_model.py \
  --language ceb \
  --model_size small \
  --audio_dir audio_data \
  --transcript_file audio_data/transcripts.txt \
  --output_dir checkpoints/ceb_whisper \
  --model_save_path models/ceb_asr_final \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --max_steps 3000
```

### 5.5 CPU-only training

```bash
python train_asr_model.py \
  --language en \
  --model_size tiny \
  --no_fp16 \
  --max_steps 500 \
  --batch_size 2
```

### 5.6 Key hyperparameters explained

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `--max_steps` | 2000 | Total training steps (increase for more data) |
| `--batch_size` | 8 | Samples per GPU step (reduce if OOM error) |
| `--learning_rate` | 1e-5 | Step size; keep low to avoid catastrophic forgetting |
| `--warmup_steps` | 100 | Steps to linearly increase LR from 0 |
| `--test_split` | 0.15 | Fraction held out for evaluation |

**Rule of thumb for `--max_steps`:**  
`max_steps = (num_samples × num_epochs) / (batch_size × grad_accum_steps)`

---

## 6. Monitoring Training

### 6.1 TensorBoard (live graphs)

```bash
# In a second terminal, while training runs
tensorboard --logdir asr_model_output/runs
# Open: http://localhost:6006
```

Watch for:
- `train/loss` → should decrease steadily
- `eval/wer` → Word Error Rate (%) — lower is better
- `eval/loss` → should decrease; if it rises while train/loss falls = overfitting

### 6.2 Console logs

Every 25 steps the script prints:
```
{'loss': 1.23, 'learning_rate': 9.5e-06, 'epoch': 0.83}
```

Every `eval_steps` (default 200):
```
{'eval_loss': 0.85, 'eval_wer': 18.4, 'eval_runtime': 12.3}
```

### 6.3 Early stopping

Training stops automatically if WER does not improve for **3 consecutive evaluations** (controlled by `EarlyStoppingCallback`). This prevents overfitting on small datasets.

---

## 7. Saving & Versioning the Model

### 7.1 Automatic saves

- **Checkpoints** are saved every `--save_steps` (default 200) inside `asr_model_output/`
- **Best checkpoint** is automatically restored at the end of training
- **Final model** is written to `asr_model_final/` (or `--model_save_path`)

### 7.2 Final model contents

```
asr_model_final/
├── config.json               ← model architecture
├── model.safetensors         ← weights
├── tokenizer.json            ← vocabulary
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── preprocessor_config.json  ← feature extractor settings
├── generation_config.json
└── training_config.json      ← snapshot of your training parameters
```

### 7.3 Resume training from a checkpoint

```bash
# HuggingFace Trainer auto-resumes if output_dir contains a checkpoint
python train_asr_model.py \
  --output_dir asr_model_output   # same dir as before
```

### 7.4 Version control recommendation

```bash
# Tag each experiment
cp -r asr_model_final models/ceb_whisper_small_v1
cp -r asr_model_final models/ceb_whisper_small_v2  # after more data
```

Or push to HuggingFace Hub:
```bash
pip install huggingface_hub
huggingface-cli login
# Then in Python:
# trainer.push_to_hub("your-username/ceb-whisper-small")
```

---

## 8. Inference

### 8.1 Transcribe a single file

```bash
python train_asr_model.py \
  --mode infer \
  --audio_path path/to/test_audio.wav \
  --model_save_path asr_model_final \
  --language ceb
```

### 8.2 Batch transcription

Create a text file listing audio paths, one per line:

```text
# audio_files.txt
audio_data/test1.wav
audio_data/test2.wav
recordings/interview.mp3
```

```bash
python train_asr_model.py \
  --mode batch_infer \
  --audio_list audio_files.txt \
  --model_save_path asr_model_final \
  --language ceb
```

Results are saved to `batch_transcription_results.csv`.

### 8.3 Inference from Python code

```python
from train_asr_model import transcribe, batch_transcribe

# Single file
text = transcribe(
    audio_path="audio_data/test.wav",
    model_path="asr_model_final",
    language="ceb",
    num_beams=5,
)
print(text)

# Multiple files
results = batch_transcribe(
    audio_paths=["test1.wav", "test2.wav"],
    model_path="asr_model_final",
    language="ceb",
)
for r in results:
    print(r["file"], "→", r["transcription"])
```

### 8.4 Using the model with HuggingFace pipeline (alternative)

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="asr_model_final",
    device=0,          # GPU index, or -1 for CPU
)

result = asr("audio_data/test.wav")
print(result["text"])
```

---

## 9. Evaluating Model Quality

### 9.1 Word Error Rate (WER)

WER is the primary metric: **lower is better**.

$$WER = \frac{S + D + I}{N} \times 100\%$$

Where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total reference words.

| WER Range | Quality |
|-----------|---------|
| < 5% | Professional / commercial quality |
| 5–15% | Usable for most applications |
| 15–30% | Acceptable for controlled environments |
| > 30% | Needs more data or larger model |

### 9.2 Run evaluation manually

```python
import evaluate
from train_asr_model import transcribe

wer_metric = evaluate.load("wer")

# Your ground-truth pairs
test_pairs = [
    ("audio_data/test1.wav", "The quick brown fox"),
    ("audio_data/test2.wav", "She sells seashells"),
]

predictions = []
references  = []

for audio_path, true_text in test_pairs:
    pred = transcribe(audio_path, "asr_model_final", language="ceb")
    predictions.append(pred.lower())
    references.append(true_text.lower())

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"WER: {wer * 100:.2f}%")
```

### 9.3 Improving a poor WER score

| Problem | Solution |
|---------|----------|
| High WER on test, low on train | Overfitting → add more data, reduce max_steps |
| High WER everywhere | Not enough data or model too small |
| WER good but sounds wrong | Increase `--num_beams` for decoding |
| Specific words wrong | Add those utterances to training set |
| Background noise confusion | Clean data or add noise-augmented samples |

---

## 10. Tips & Troubleshooting

### Out of memory (OOM) on GPU

```bash
# Reduce batch size and increase gradient accumulation to compensate
python train_asr_model.py --batch_size 2 --model_size tiny
```

Also set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` in your shell.

### Apple Silicon (M1/M2/M3) — MPS backend

```bash
python train_asr_model.py --no_fp16
```

MPS does not support fp16 in all ops. The script will automatically use CPU or bf16 when available.

### `libsndfile` / `soundfile` error

```bash
brew install libsndfile   # macOS
# or
pip install soundfile --force-reinstall
```

### `evaluate` / `jiwer` version conflict

```bash
pip install jiwer==3.0.3 evaluate==0.4.1
```

### Training is very slow on CPU

Use `--model_size tiny` and limit steps:

```bash
python train_asr_model.py --model_size tiny --max_steps 100 --no_fp16 --batch_size 2
```

### Dataset too small warning

With fewer than ~30 samples the 15% test split may have too few samples for reliable WER. Either:
- Collect more data (recommended)
- Reduce `--test_split` to `0.1` temporarily

---

## 11. Language Code Reference

Use the BCP-47 code with `--language`:

| Language | Code |
|----------|------|
| English | `en` |
| Cebuano | `ceb` |
| Filipino / Tagalog | `tl` |
| Spanish | `es` |
| French | `fr` |
| Mandarin Chinese | `zh` |
| Japanese | `ja` |
| Korean | `ko` |
| Arabic | `ar` |
| Hindi | `hi` |
| Portuguese | `pt` |
| German | `de` |
| Italian | `it` |
| Russian | `ru` |
| Vietnamese | `vi` |
| Indonesian | `id` |
| Malay | `ms` |

> Full list of Whisper-supported languages:  
> https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

---

## 12. Repository Scope, License, and Legal Disclaimers

This repository is the **mainline for Davao Cebuano ASR only** and is configured for **Whisper-based training workflows**.

### 12.1 Scope policy (fork required for out-of-scope work)

- If you want to work on **other Philippine languages** in a dedicated way, create a fork and maintain the language-specific work there.
- If you want to use an **ASR model family other than Whisper**, create a fork and maintain model-specific changes there.

### 12.2 Dataset use policy

- Contributor audio and transcripts in this project are intended for **free research and educational use**.
- **Commercial use is not allowed** for contributed dataset material under the repository data policy.
- **Public voice cloning, speaker impersonation, or synthetic voice generation of contributed speakers is prohibited**.
- See repository policy files for full terms:
  - `README.md`
  - `CONTRIBUTING.md`
  - `DATA_LICENSE.md`

### 12.3 Privacy and legal compliance

- Contributors and maintainers must comply with Philippine privacy law, including the **Data Privacy Act of 2012 (Republic Act No. 10173)** and related regulations.
- Data collection must be based on informed consent, lawful processing, and responsible handling of potentially identifiable voice data.
- See:
  - `PRIVACY_AND_COMPLIANCE.md`
  - `Voice_Recording_Transcript_Consent_Form.docx`

### 12.4 Important disclaimer

These materials are provided for project governance and risk reduction and are **not legal advice**. For institutional deployment or publication, obtain review from your legal/compliance office.

### 12.5 Citation for research papers

- If you use this repository in published work, cite it using `CITATION.cff`.
- A ready-to-copy BibTeX entry is also provided in `README.md`.

---

## Quick Reference Card

```bash
# Install dependencies
pip install transformers datasets accelerate evaluate jiwer librosa tensorboard soundfile

# Train (English, Whisper-small, default paths)
python train_asr_model.py

# Train (Cebuano, Whisper-base)
python train_asr_model.py --language ceb --model_size base

# Monitor
tensorboard --logdir asr_model_output/runs

# Transcribe one file
python train_asr_model.py --mode infer --audio_path my_audio.wav --language ceb

# Transcribe many files
python train_asr_model.py --mode batch_infer --audio_list files.txt --language ceb
```
