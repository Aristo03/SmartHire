import io
from pathlib import Path

import docx   # python-docx
import fitz   # PyMuPDF


def extract_text_from_pdf(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:              # many resumes put skills in tables
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def extract_text(data: bytes, filename: str) -> str:
    """Extract plain text from an uploaded resume (PDF, DOCX or TXT)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(data)
    elif ext == ".docx":
        text = extract_text_from_docx(data)
    elif ext == ".txt":
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload a PDF, DOCX or TXT file.")

    if not text.strip():
        raise ValueError("No text found. The file may be a scanned image (OCR is not supported).")
    return text