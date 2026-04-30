#!/usr/bin/env python3
"""
Test a fine-tuned Whisper ASR model from a saved checkpoint.

Supports:
1) Real-time chunked microphone transcription
2) Transcription from a specified audio file
3) Structured minimum test-case rounds with CER/WER logging

Usage examples:
  python test_asr_model.py
  python test_asr_model.py --mode realtime --chunk-duration 4
  python test_asr_model.py --mode file --audio-path audio_data/sample.wav
    python test_asr_model.py --model-path asr_model_final --language tl
    python test_asr_model.py --mode testcases
    python test_asr_model.py --mode testcases --test-case-file asr_minimum_test_cases_tl.txt
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import jiwer
import librosa
import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WHISPER_SR = 16_000
MIN_ACCEPTABLE_TEST_CASES = 20
DEFAULT_TEST_CASE_FILE = "asr_minimum_test_cases_tl.txt"
DEFAULT_TEST_OUTPUT_DIR = "asr_test_results"


def normalize_text(text: str) -> str:
    """Normalize text for consistent CER/WER scoring."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_test_cases(test_case_file: str) -> list[str]:
    """Load test cases from text file (one sentence per line)."""
    path = Path(test_case_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Test case file not found: {path}. "
            "Create it with at least 20 test sentences."
        )

    cases = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) < MIN_ACCEPTABLE_TEST_CASES:
        raise ValueError(
            f"Test case file must contain at least {MIN_ACCEPTABLE_TEST_CASES} cases. "
            f"Found: {len(cases)}"
        )
    return cases


def resolve_model_path(user_path: Optional[str]) -> Path:
    """Resolve the model path, preferring user-provided path when available."""
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        return p

    # Prefer asr_model_final (user request), then common alternate location.
    candidates = [
        Path("asr_model_final"),
        Path("asr_model_output/asr_model_final"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not find a saved model. Expected one of: asr_model_final, "
        "asr_model_output/asr_model_final"
    )


class ASRTester:
    def __init__(self, model_path: Path, language: str = "tl", num_beams: int = 5):
        self.model_path = model_path
        self.language = language
        self.num_beams = num_beams

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Loading model from: {self.model_path}")
        print(f"[Init] Device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def record_chunk(self, duration: float, input_device: Optional[int] = None) -> np.ndarray:
        """Record a single audio chunk from microphone."""
        audio = sd.rec(
            int(duration * WHISPER_SR),
            samplerate=WHISPER_SR,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
        return audio.squeeze()

    def transcribe_array(self, audio_array: np.ndarray) -> str:
        """Transcribe a mono audio array sampled at 16 kHz."""
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task="transcribe",
        )

        inputs = self.processor.feature_extractor(
            audio_array,
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
        ).input_features.to(self.device)

        with torch.no_grad():
            pred_ids = self.model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=self.num_beams,
            )

        text = self.processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )[0].strip()
        return text

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a single audio file path."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio, _ = librosa.load(str(path), sr=WHISPER_SR, mono=True)
        text = self.transcribe_array(audio)
        return text

    def transcribe_realtime(self, chunk_duration: float = 5.0, input_device: Optional[int] = None):
        """
        Repeatedly record fixed-duration chunks from microphone and transcribe each chunk.
        Stop with Ctrl+C.
        """
        print("\n[Realtime] Starting real-time chunked transcription")
        print(f"[Realtime] Chunk duration: {chunk_duration:.1f}s")
        if input_device is not None:
            print(f"[Realtime] Input device id: {input_device}")
        print("[Realtime] Press Ctrl+C to stop\n")

        try:
            while True:
                print("[Realtime] Listening...")
                audio = self.record_chunk(duration=chunk_duration, input_device=input_device)

                text = self.transcribe_array(audio)
                if text:
                    print(f"[Realtime] Transcription: {text}\n")
                else:
                    print("[Realtime] (No transcription)\n")

        except KeyboardInterrupt:
            print("\n[Realtime] Stopped by user.")

    def run_minimum_test_round(
        self,
        test_cases: list[str],
        chunk_duration: float = 5.0,
        input_device: Optional[int] = None,
        output_dir: str = DEFAULT_TEST_OUTPUT_DIR,
    ) -> Optional[Path]:
        """
        Run a structured test round and save CER/WER only if all cases are completed.

        Returns the detailed result CSV path if saved; otherwise None.
        """
        total_cases = len(test_cases)
        print("\n[Round] Structured ASR test round")
        print(f"[Round] Total required test cases: {total_cases}")
        print(f"[Round] Minimum acceptable test cases: {MIN_ACCEPTABLE_TEST_CASES}")
        print("[Round] You must complete every test case. CER/WER is not saved on early exit.\n")

        rows: list[dict] = []

        for i, reference in enumerate(test_cases, start=1):
            print(f"\n[Case {i}/{total_cases}] Read this prompt:")
            print(f"  {reference}")

            while True:
                action = input("Press Enter to record, 'r' to re-show prompt, or 'q' to abort round: ").strip().lower()
                if action == "q":
                    print("[Round] Aborted before completion. CER/WER was not saved.")
                    return None
                if action == "r":
                    print(f"  {reference}")
                    continue

                print("[Round] Recording...")
                audio = self.record_chunk(duration=chunk_duration, input_device=input_device)
                hypothesis = self.transcribe_array(audio)

                ref_norm = normalize_text(reference)
                hyp_norm = normalize_text(hypothesis)
                case_cer = 100.0 * jiwer.cer(ref_norm, hyp_norm)
                case_wer = 100.0 * jiwer.wer(ref_norm, hyp_norm)

                print(f"[Case {i}] ASR output: {hypothesis}")
                print(f"[Case {i}] CER: {case_cer:.2f}%")
                print(f"[Case {i}] WER: {case_wer:.2f}%")

                accept = input("Accept this attempt? [y = yes, n = retry, q = abort round]: ").strip().lower()
                if accept == "q":
                    print("[Round] Aborted before completion. CER/WER was not saved.")
                    return None
                if accept == "y":
                    rows.append(
                        {
                            "case_index": i,
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "reference_normalized": ref_norm,
                            "hypothesis_normalized": hyp_norm,
                            "case_cer_percent": round(case_cer, 4),
                            "case_wer_percent": round(case_wer, 4),
                        }
                    )
                    break

        # Save only after all test cases are completed.
        references = [row["reference_normalized"] for row in rows]
        hypotheses = [row["hypothesis_normalized"] for row in rows]
        corpus_cer = 100.0 * jiwer.cer(references, hypotheses)
        corpus_wer = 100.0 * jiwer.wer(references, hypotheses)
        mean_case_cer = float(np.mean([row["case_cer_percent"] for row in rows]))
        mean_case_wer = float(np.mean([row["case_wer_percent"] for row in rows]))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        details_path = out_dir / f"asr_test_round_{ts}.csv"
        with open(details_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_index",
                    "reference",
                    "hypothesis",
                    "reference_normalized",
                    "hypothesis_normalized",
                    "case_cer_percent",
                    "case_wer_percent",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_path = out_dir / "asr_test_rounds_summary.csv"
        summary_exists = summary_path.exists()
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round_timestamp",
                    "model_path",
                    "language",
                    "num_cases",
                    "mean_case_cer_percent",
                    "mean_case_wer_percent",
                    "corpus_cer_percent",
                    "corpus_wer_percent",
                    "details_csv",
                ],
            )
            if not summary_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "round_timestamp": ts,
                    "model_path": str(self.model_path),
                    "language": self.language,
                    "num_cases": len(rows),
                    "mean_case_cer_percent": round(mean_case_cer, 4),
                    "mean_case_wer_percent": round(mean_case_wer, 4),
                    "corpus_cer_percent": round(corpus_cer, 4),
                    "corpus_wer_percent": round(corpus_wer, 4),
                    "details_csv": str(details_path),
                }
            )

        print("\n[Round] Completed all required test cases.")
        print(f"[Round] Mean case CER: {mean_case_cer:.2f}%")
        print(f"[Round] Mean case WER: {mean_case_wer:.2f}%")
        print(f"[Round] Corpus CER: {corpus_cer:.2f}%")
        print(f"[Round] Corpus WER: {corpus_wer:.2f}%")
        print(f"[Round] Detailed results saved: {details_path}")
        print(f"[Round] Summary log saved: {summary_path}")
        return details_path


def interactive_menu(tester: ASRTester):
    """Interactive mode that lets the user choose how to transcribe."""
    while True:
        print("\n=== ASR Model Tester ===")
        print("1. Real-time microphone transcription")
        print("2. Transcribe an audio file")
        print("3. Run minimum test-case round and compute CER + WER")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            chunk_raw = input("Chunk duration in seconds [5]: ").strip()
            chunk_duration = float(chunk_raw) if chunk_raw else 5.0

            dev_raw = input("Input device id (optional): ").strip()
            input_device = int(dev_raw) if dev_raw else None

            tester.transcribe_realtime(
                chunk_duration=chunk_duration,
                input_device=input_device,
            )

        elif choice == "2":
            audio_path = input("Audio file path: ").strip()
            if not audio_path:
                print("No audio path provided.")
                continue

            try:
                text = tester.transcribe_file(audio_path)
                print(f"\n[File] Transcription:\n{text}\n")
            except Exception as exc:
                print(f"[File] Error: {exc}")

        elif choice == "3":
            test_case_file = input(
                f"Test case file path [{DEFAULT_TEST_CASE_FILE}]: "
            ).strip() or DEFAULT_TEST_CASE_FILE

            chunk_raw = input("Chunk duration in seconds [5]: ").strip()
            chunk_duration = float(chunk_raw) if chunk_raw else 5.0

            dev_raw = input("Input device id (optional): ").strip()
            input_device = int(dev_raw) if dev_raw else None

            out_dir = input(
                f"Output directory for CER/WER logs [{DEFAULT_TEST_OUTPUT_DIR}]: "
            ).strip() or DEFAULT_TEST_OUTPUT_DIR

            try:
                cases = load_test_cases(test_case_file)
                tester.run_minimum_test_round(
                    test_cases=cases,
                    chunk_duration=chunk_duration,
                    input_device=input_device,
                    output_dir=out_dir,
                )
            except Exception as exc:
                print(f"[Round] Error: {exc}")

        elif choice == "4":
            print("Goodbye.")
            return

        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a saved ASR model with real-time or file transcription modes."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "realtime", "file", "testcases"],
        default="interactive",
        help="Testing mode (default: interactive)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to saved model directory (default: auto-detect asr_model_final)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Audio path for file mode",
    )
    parser.add_argument(
        "--language",
        default="tl",
        help="BCP-47 language code for decoding prompt (default: tl)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam width for decoding (default: 5)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Chunk duration in seconds for realtime mode (default: 5)",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Optional sounddevice input device id for realtime mode",
    )
    parser.add_argument(
        "--test-case-file",
        default=DEFAULT_TEST_CASE_FILE,
        help=(
            "Path to test-case text file (one sentence per line). "
            f"Must contain at least {MIN_ACCEPTABLE_TEST_CASES} cases."
        ),
    )
    parser.add_argument(
        "--test-output-dir",
        default=DEFAULT_TEST_OUTPUT_DIR,
        help="Directory where round-level CER/WER logs are saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    tester = ASRTester(
        model_path=model_path,
        language=args.language,
        num_beams=args.num_beams,
    )

    if args.mode == "interactive":
        interactive_menu(tester)

    elif args.mode == "realtime":
        tester.transcribe_realtime(
            chunk_duration=args.chunk_duration,
            input_device=args.input_device,
        )

    elif args.mode == "file":
        if not args.audio_path:
            raise ValueError("--audio-path is required when --mode file")
        text = tester.transcribe_file(args.audio_path)
        print(f"\n[File] Transcription:\n{text}\n")

    elif args.mode == "testcases":
        cases = load_test_cases(args.test_case_file)
        tester.run_minimum_test_round(
            test_cases=cases,
            chunk_duration=args.chunk_duration,
            input_device=args.input_device,
            output_dir=args.test_output_dir,
        )


if __name__ == "__main__":
    main()
