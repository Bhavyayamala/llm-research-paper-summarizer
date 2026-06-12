from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = """
Artificial Intelligence is transforming the world.
Machine learning and deep learning are important
parts of AI. These technologies are used in
healthcare, education, finance and many other
domains.
"""

summary = summarizer(
    text,
    max_length=50,
    min_length=20,
    do_sample=False
)

print(summary[0]["summary_text"])