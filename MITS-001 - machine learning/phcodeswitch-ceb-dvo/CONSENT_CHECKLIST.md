# Consent Checklist

Use this checklist before accepting contributor audio/transcript data.

## For Contributors

Complete all items:
- Read README.md, CONTRIBUTING.md, DATA_LICENSE.md, and PRIVACY_AND_COMPLIANCE.md.
- Review Voice_Recording_Transcript_Consent_Form.docx.
- Confirm understanding that data is for free non-commercial research/education use only.
- Confirm understanding that commercial use is prohibited in this repository.
- Confirm understanding that public voice cloning, speaker impersonation, or synthetic voice generation of contributed speakers is prohibited.
- Confirm understanding of repository scope: Davao Cebuano + Whisper in mainline.
- Confirm understanding that other languages or non-Whisper model work belongs in a fork.
- Sign and date the consent form.

## For Maintainers

Complete all items before merge/release:
- Verify signed consent form exists for each contributor.
- Verify contributor accepted non-commercial data license terms.
- Verify contributor accepted the no-public-voice-cloning restriction.
- Verify contributor accepted privacy-law compliance terms.
- Verify contribution is in scope (Davao Cebuano + Whisper).
- If out of scope, redirect contributor to a fork workflow.

## Consent Form Update Utility

If you need to regenerate the consent form document with repository clauses:
- Run: python scripts/rewrite_consent_doc.py
