import io

import docx
import fitz
import pytest

from src.features.match_features import extract_skills
from src.parsing.resume_parser import extract_text

SAMPLE = "Data Scientist skilled in Python, SQL and machine learning."


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


def make_docx(text: str) -> bytes:
    document = docx.Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_pdf_extraction():
    assert "Python" in extract_text(make_pdf(SAMPLE), "cv.pdf")


def test_docx_extraction():
    assert "Python" in extract_text(make_docx(SAMPLE), "cv.docx")


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        extract_text(b"fake image bytes", "cv.png")


def test_empty_file():
    with pytest.raises(ValueError):
        extract_text(b"   ", "cv.txt")


def test_skill_extraction():
    assert extract_skills(SAMPLE) == ["machine learning", "python", "sql"]


def test_skill_whole_word_only():
    # 'java' must NOT be found inside 'javascript'
    assert extract_skills("I build apps in JavaScript") == ["javascript"]