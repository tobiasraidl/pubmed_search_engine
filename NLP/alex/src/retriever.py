import numpy as np
import json
import pandas as pd
import torch
from tqdm import tqdm
from bioasq_embedding_model import BioASQEmbeddingModel
from sentence_transformers import SentenceTransformer


class BioASQEmbeddingRetriever:
    def __init__(self):
        self.model = BioASQEmbeddingModel()
        self.document_embeddings_path = "../out/embeddings/document_embeddings_BioASQEmbeddingModel.json"
        self.df_embeddings = self._load_document_embeddings(
            path=self.document_embeddings_path
        )

    def _load_document_embeddings(self, path) -> pd.DataFrame:
        """
        Loads url and document embeddings from JSON and returns dataframe.
        """
        with open(path, "r") as file:
            document_embeddings = json.load(file)
        return pd.DataFrame(document_embeddings).dropna()

    def cosine_similarity(self, query, vectors, n=10):
        """
        Calculate cosine sim. given a query against a set of vectors.
        Returns indices (top n similar), similarity_scores
        """
        # Normalize the query and vectors
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(vectors_norm, query_norm)

        # Get indices of top 5 most similar vectors
        top_indices = np.argsort(similarities)[-n:][::-1]
        return top_indices, similarities[top_indices]

    def retrieve(self, query: str, n=10, return_df=False) -> pd.DataFrame:
        query_embedding = self.model.encode(query)
        document_embeddings = np.vstack(self.df_embeddings["document_embedding"].values)

        top_indices, similarities = self.cosine_similarity(
            query_embedding, document_embeddings, n=10
        )
        result_df = self.df_embeddings.loc[top_indices, ["url"]].copy()
        result_df["cosine_similarity"] = similarities
        
        if return_df:
            return result_df.reset_index(drop=True)
        else:
            return [{"url": row["url"], "cosine_similarity": row["cosine_similarity"]}for _, row in result_df.iterrows()]


    def batch_retrieve(self, queries, n=10):
        """
        Retrieves a list of documents for each query. (list)

        Args:
            queries (list): List of queries to retrieve documents for.
            n (int): Number of documents returned for each query.

        Returns:
            (list): A list of dicts, where each dict contains the query and the top n documents for this query.
        """
        results = []
        for query in tqdm(queries, desc="Preprocessing queries", unit="queries"):
            query_documents = self.retrieve(query, n=n)
            query_documents = [doc["url"] for doc in query_documents]
            results.append({"body": query, "documents": query_documents})
        return results

class BertRetriever:
    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", model_name="all-MiniLM-L6-v2"):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        self.document_embeddings_path = f"../out/embeddings/document_embeddings_{model_name}.json"
        self.df_embeddings = self._load_document_embeddings(
            path=self.document_embeddings_path
        )

    def _load_document_embeddings(self, path) -> pd.DataFrame:
        """
        Loads url and document embeddings from JSON and returns dataframe.
        """
        with open(path, "r") as file:
            document_embeddings = json.load(file)
        return pd.DataFrame(document_embeddings).dropna()
    
    def cosine_similarity(self, query, vectors, n=10):
        """
        Calculate cosine sim. given a query against a set of vectors.
        Returns indices (top n similar), similarity_scores
        """
        # Normalize the query and vectors
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(vectors_norm, query_norm)

        # Get indices of top 5 most similar vectors
        top_indices = np.argsort(similarities)[-n:][::-1]
        return top_indices, similarities[top_indices]

    def retrieve(self, query: str, n=10, return_df=False) -> dict:
        query_embedding = self.model.encode(query)
        document_embeddings = np.vstack(self.df_embeddings["document_embedding"].values)

        top_indices, similarities = self.cosine_similarity(
            query_embedding, document_embeddings, n=10
        )
        result_df = self.df_embeddings.loc[top_indices, ["url"]].copy()
        result_df["cosine_similarity"] = similarities
        
        if return_df:
            return result_df.reset_index(drop=True)
        else:
            return [{"url": row["url"], "cosine_similarity": row["cosine_similarity"]}for _, row in result_df.iterrows()]

    def batch_retrieve(self, queries, n=10):
        """
        Retrieves a list of documents for each query. (list)

        Args:
            queries (list): List of queries to retrieve documents for.
            n (int): Number of documents returned for each query.

        Returns:
            (list): A list of dicts, where each dict contains the query and the top n documents for this query.
        """
        results = []
        for query in tqdm(queries, desc="Retrieving documents for each query", unit="queries"):
            query_documents = self.retrieve(query, n=n)
            query_documents = [doc["url"] for doc in query_documents]
            results.append({"body": query, "documents": query_documents})
        return results
    

if __name__ == "__main__":
    # example code
    retriever = BioASQEmbeddingRetriever()
    result_df = retriever.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10, return_df=True)
    print(result_df)

    # test batch retrieve
    r = retriever.batch_retrieve(["Test", "Hello World", "Alex"], n=10)
    r

    retriever = BertRetriever()
    result_df = retriever.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10, return_df=True)

    