from pathlib import Path

from docx import Document

path = Path(__file__).resolve().parents[1] / "Voice_Recording_Transcript_Consent_Form.docx"
doc = Document(path)
for i, para in enumerate(doc.paragraphs, 1):
    text = para.text.strip()
    if text:
        print(f"{i}: {text}")
