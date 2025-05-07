from sentence_transformers import SentenceTransformer
import json
from pinecone import Pinecone, ServerlessSpec
import torch
from tqdm import tqdm


# Load model and move to GPU if available
model = SentenceTransformer('reranker/out/model_1')
if torch.cuda.is_available():
    model = model.to('cuda')


# Example list of documents
with open("data/train/documents.json", "r") as f:
    documents = json.load(f)

# Get the text of the documents and create embeddings
doc_texts = [f"{doc['title']} {doc['abstract']}" for doc in documents]
doc_embeddings = model.encode(doc_texts)  # Assume the model is already loaded
print("Embeddings generated")
# Extract the document Iurls
doc_urls = [doc["url"] for doc in documents]


# Initialize Pinecone client
pc = Pinecone(api_key="set ur own")

# Check if the index already exists and create it if necessary
if 'air25' not in pc.list_indexes().names():
    pc.create_index(
        name='air25',
        dimension=384,  # The dimension of your embeddings (e.g., 512 or 768 for BERT-based models)
        metric='cosine',  # Metric for distance calculation (use 'cosine' for cosine similarity)
        spec=ServerlessSpec(  # Define the environment settings for your index
            cloud='aws',
            region='us-east-1'  # Choose the appropriate region
        )
    )
    
# Initialize Pinecone index
index = pc.Index('air25')

# Prepare data to upsert: Pair each document url with its corresponding embedding
# upsert_data = [{"id": doc_urls[i], "values": doc_embeddings[i]} for i in range(len(documents))]

batch_size = 100
upsert_data = [] # list of batches (list of lists of dicts)
for i in range(0, len(documents), batch_size):
    batch_embeddings = doc_embeddings[i:i + batch_size]
    batch_urls = doc_urls[i:i + batch_size]
    batch_upsert_data = [{"id": url, "values": embedding} for url, embedding in zip(batch_urls, batch_embeddings)]
    upsert_data.append(batch_upsert_data)
    index.upsert(vectors=batch_upsert_data)


print("Data successfully upserted into Pinecone!")