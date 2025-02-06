import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "This is a sample sentence.",
    "Another sentence for embedding.",
    "More sentences to test the model.",
    "Creating embeddings for this text.",
    "We are testing FAISS integration."
]

embeddings = model.encode(sentences)

embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

query = "sample sentence"
query_embedding = model.encode([query]).astype('float32')
k = 2  
distances, indices = index.search(query_embedding, k)

print(f"Query: {query}")
print("Nearest Neighbors:")
for i in range(k):
    print(f"Sentence: {sentences[indices[0][i]]}, Distance: {distances[0][i]}")
