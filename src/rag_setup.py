import fitz
import chromadb

from sentence_transformers import SentenceTransformer
from chunker import split_text
from text_cleaner import clean_text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
pdf_path = "data/sample_paper.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

doc.close()

# Clean text
text = clean_text(text)

# Chunk text
chunks = split_text(text)

print("Chunks:", len(chunks))

# Create ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Delete old collection if exists
try:
    client.delete_collection("research_papers")
except:
    pass

collection = client.create_collection(
    name="research_papers"
)

# Store embeddings
for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"chunk_id": i}]
    )

print("Embeddings stored successfully!")