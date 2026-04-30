# phcodeswitch-ceb-dvo

Main repository for fine-tuning a Whisper ASR model for code-switching scenarios involving English and Davao Cebuano.

## Project Scope

This repository is intentionally scoped to:
- Davao Cebuano dataset development and training workflows.
- Whisper-based ASR training and inference workflows.

If you want to work on other Philippine languages or non-Whisper model families, create and maintain a fork.

## Contributor Requirements

Before contributing any audio/transcript data, contributors must:
- Read and accept repository policy documents.
- Review and sign the consent form before data contribution.
- Agree that contributed data may be shared for free non-commercial research and educational use.
- Agree that commercial use is prohibited for contributed dataset material in this repository.
- Agree that public voice cloning, speaker impersonation, or voice synthesis of contributed speakers is prohibited.
- Agree to Philippine data privacy compliance requirements.

See:
- CONTRIBUTING.md
- DATA_LICENSE.md
- PRIVACY_AND_COMPLIANCE.md
- CONSENT_CHECKLIST.md
- Voice_Recording_Transcript_Consent_Form.docx

## Legal and Compliance Notes

- This repository includes policy and disclaimer documents to support informed participation and responsible data handling.
- These materials are not legal advice. Seek institutional legal review when needed.

## How to Cite

Use the citation metadata in `CITATION.cff`.

BibTeX:

```bibtex
@misc{emberda_phcodeswitch_ceb_dvo_2026,
	title        = {phcodeswitch-ceb-dvo: Fine-tuning an Automatic Speech Recognition Model for Davao-Cebuano with Code-Switching Support},
	author       = {Emberda, Eric John},
	year         = {2026},
	howpublished = {GitHub repository},
	note         = {Accessed: 2026-04-18},
	url          = {https://github.com/ericjohnemberda/phcodeswitch-ceb-dvo}
}
```

## Model Testing

Use test_asr_model.py to test the fine-tuned model from asr_model_final.

- Interactive menu (choose real-time microphone or audio file mode):
	python test_asr_model.py
- Real-time chunked transcription directly:
	python test_asr_model.py --mode realtime --chunk-duration 5
- Transcribe a specific audio file directly:
	python test_asr_model.py --mode file --audio-path path/to/audio.wav
- Run structured minimum test-case round with CER + WER logging:
	python test_asr_model.py --mode testcases

Useful optional flags:
- --model-path (override model location)
- --language (default: tl)
- --num-beams (default: 5)
- --input-device (optional microphone device id for real-time mode)
- --test-case-file (default: asr_minimum_test_cases_tl.txt; minimum 20 cases)
- --test-output-dir (default: asr_test_results)

CER and WER outputs are saved only after all required test cases are completed:
- Round details: asr_test_results/asr_test_round_YYYYMMDD_HHMMSS.csv
- Round summary: asr_test_results/asr_test_rounds_summary.csv

## Training and Usage Guide

For setup and model training instructions, see ASR_TRAINING_GUIDE.md.
