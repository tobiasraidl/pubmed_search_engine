from sentence_transformers import SentenceTransformer
import json
from pinecone import Pinecone, ServerlessSpec



def query_pinecone(query, top_k=10):
    # Load the trained model
    model = SentenceTransformer('reranker/out/model_1')

    query_embedding = model.encode(query).tolist()

    pc = Pinecone(api_key="set your own")
    index = pc.Index("air25")

    response = index.query(
        # namespace="ns1",
        vector=query_embedding,
        top_k=top_k,
        # include_values=True,
        include_metadata=True,
        # filter={"genre": {"$eq": "action"}}
    )
    
    return response["matches"]

if __name__ == "__main__":
    query = "Modified Vaccinia Ankara Virus Vaccination Provides Long-Term Protection against Nasal Rabbitpox Virus Challenge"
    response = query_pinecone(query, top_k=10)
    print(response)
    
    
