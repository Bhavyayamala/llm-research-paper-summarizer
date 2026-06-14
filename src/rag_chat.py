import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    "research_papers"
)

while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    # Create query embedding
    query_embedding = embed_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances"]
    )

    print("\nRetrieved IDs:")
    print(results["ids"])

    print("\nDistances:")
    print(results["distances"])

    # Join all retrieved chunks
    context = "\n".join(
        results["documents"][0][:3])

    context = context[:1200]
    

    print("\nNumber of retrieved chunks:")
    print(len(results["documents"][0]))

    print("\nRetrieved Context Preview:")
    print(context[:1000])

    context_length=len(context)
    print("Context length:", context_length)

    # Prompt
    prompt = f"""
You are an expert research paper assistant.

Use ONLY the provided context.

Answer the question clearly in 2-4 sentences.

If the information is not present in the context, reply:
"Information not found in retrieved context."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False
    )

    print("\nAnswer:")
    print(answer[0]["generated_text"])