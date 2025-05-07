import requests
import json
import time
from tqdm import tqdm

with open("data/train/documents.json", "r") as f:
    documents = json.load(f)

document_bodies = [f"{doc['title']} {doc['abstract']}" for doc in documents]

def split_into_n_chunks(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

document_batches = split_into_n_chunks(document_bodies, 1000)

# Set your Hugging Face API Token
HUGGINGFACE_TOKEN = "set your own"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
}

# Define the model you want to use
model_name = "sentence-transformers/all-mpnet-base-v2"

# Set up the API URL for inference
url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

embeddings = []

# Send the POST request to the Hugging Face API
for i, document_batch in tqdm(enumerate(document_batches), desc="Processing batches"):
    response = requests.post(url, headers=headers, json={"inputs": document_batch})

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        print(f"Stopped at batch {i}/{len(document_batches)}")
        break

    # Extract the embeddings from the response
    batch_embeddings = response.json()
    embeddings.extend(batch_embeddings)
    time.sleep(5)
    
for i, document in enumerate(documents):
    documents[i]["embedding"] = embeddings[i]
    
with open("reranker/embeddings.json", "w") as f:
    json.dump(documents, f, indent=2)
    
