import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Sample list of sentences (can be your text data from the previous step)
sentences = [
    "This is a sample sentence.",
    "Another sentence for embedding.",
    "More sentences to test the model.",
    "Creating embeddings for this text.",
    "We are testing FAISS integration."
]

# Step 3: Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Step 4: Convert embeddings to numpy arrays (FAISS uses numpy arrays)
embeddings = np.array(embeddings).astype('float32')

# Step 5: Create the FAISS index
dimension = embeddings.shape[1]  # The dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)

# Step 6: Add the embeddings to the FAISS index
index.add(embeddings)

# Step 7: Perform a similarity search
query = "sample sentence"
query_embedding = model.encode([query]).astype('float32')

# Step 8: Search the index for the nearest neighbors
k = 2  # Number of closest results to retrieve
distances, indices = index.search(query_embedding, k)

# Print the results
print(f"Query: {query}")
print("Nearest Neighbors:")
for i in range(k):
    print(f"Sentence: {sentences[indices[0][i]]}, Distance: {distances[0][i]}")
