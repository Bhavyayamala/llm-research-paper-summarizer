from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

context = """
Zero-shot approaches consistently fail to outperform
simple baselines and perform particularly poorly on
negative stock movement prediction.
"""

question = "What are the limitations of this paper?"

prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

result = generator(
    prompt,
    max_new_tokens=100
)

print(result[0]["generated_text"])