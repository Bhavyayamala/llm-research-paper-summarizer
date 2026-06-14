from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence1 = "Machine learning is amazing"
sentence2 = "Artificial intelligence is fascinating"
sentence3 = "I like cricket"

emb1 = model.encode(sentence1, convert_to_tensor=True)
emb2 = model.encode(sentence2, convert_to_tensor=True)
emb3 = model.encode(sentence3, convert_to_tensor=True)

print("Similarity 1-2:", cos_sim(emb1, emb2))
print("Similarity 1-3:", cos_sim(emb1, emb3))