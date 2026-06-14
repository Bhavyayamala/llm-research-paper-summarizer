import chromadb

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection("research_papers")

question = input("Ask a question: ")

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nMost Relevant Chunks:\n")

for doc in results["documents"][0]:
    print(doc)
    print("\n" + "-"*50 + "\n")