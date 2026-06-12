import fitz
from transformers import pipeline
from chunker import split_text

# Load model
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

# Split into chunks
chunks = split_text(text, chunk_size=1000)

print("Total Chunks:", len(chunks))

all_summaries = []

for i, chunk in enumerate(chunks):

    print(f"Summarizing chunk {i+1}/{len(chunks)}...")

    summary = summarizer(
        chunk,
        max_length=100,
        min_length=30,
        do_sample=False
    )

    all_summaries.append(summary[0]["summary_text"])

final_summary = "\n".join(all_summaries)

print("\n===== FINAL SUMMARY =====\n")
print(final_summary)