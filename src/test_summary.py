from transformers import pipeline

print("Loading model...")

summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)

print("Model loaded!")

text = """
Artificial Intelligence is transforming the world.
Machine learning and deep learning are important
parts of AI. These technologies are used in healthcare,
education, finance, and many other domains.
"""

summary = summarizer(
    text,
    max_length=50,
    min_length=20,
    do_sample=False
)

print("\nSummary:")
print(summary[0]["summary_text"])