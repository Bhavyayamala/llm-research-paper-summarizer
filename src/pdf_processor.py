import fitz
from text_cleaner import clean_text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text

pdf_path = "data/sample_paper.pdf"

raw_text = extract_text_from_pdf(pdf_path)

cleaned_text = clean_text(raw_text)

print(cleaned_text[:3000])