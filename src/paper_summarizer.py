import fitz
from transformers import pipeline

# Load summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Read PDF
pdf_path = "data/sample_paper.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

doc.close()

# Use first 2000 characters
text = text[:2000]

summary = summarizer(
    text,
    max_length=150,
    min_length=50,
    do_sample=False
)

print("\n===== SUMMARY =====\n")
print(summary[0]["summary_text"])