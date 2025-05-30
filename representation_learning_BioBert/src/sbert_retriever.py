# nlp_ir/src/sbert_retriever.py
import json
import numpy as np
import pandas as pd
from sentence_embedder import SentenceEmbedder

"""
The class `SBERTRetriever` retrieves documents based on a query using cosine similarity. 
It loads pre-computed document embeddings from a JSON file and uses a Sentence-BERT model to encode the query. 
The top N most similar documents are returned along with their similarity scores. 
The cosine similarity is computed between the query vector and the document matrix, and the results are sorted to find the most relevant documents.
"""

class SBERTRetriever:
    def __init__(self, model_path="nlp_ir/model/sbert-finetuned-bioasq", emb_path="nlp_ir/data/sbert_document_embeddings.json"):
        self.embedder = SentenceEmbedder(model_path)
        with open(emb_path) as f:
            self.df = pd.DataFrame(json.load(f))

    def cosine_similarity(self, query_vec, doc_matrix):
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norm = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        sims = np.dot(doc_norm, query_norm)
        return sims

    def retrieve(self, query, top_n=10):
        q_vec = self.embedder.encode_single(query)
        docs_matrix = np.vstack(self.df["document_embedding"].values)
        sims = self.cosine_similarity(q_vec, docs_matrix)
        top_indices = np.argsort(sims)[-top_n:][::-1]

        result = self.df.iloc[top_indices][["url"]].copy()
        result["score"] = sims[top_indices]
        return result.reset_index(drop=True)

if __name__ == "__main__":
    r = SBERTRetriever()
    result = r.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", top_n=5)
    print(result)
