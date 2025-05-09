import numpy as np
import json
import pandas as pd
import torch
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

    def retrieve(self, query: str, n=10) -> pd.DataFrame:
        query_embedding = self.model.encode(query)
        document_embeddings = np.vstack(self.df_embeddings["document_embedding"].values)

        top_indices, similarities = self.cosine_similarity(
            query_embedding, document_embeddings, n=10
        )
        result_df = self.df_embeddings.loc[top_indices, ["url"]].copy()
        result_df["cosine_similarity"] = similarities
        return result_df.reset_index(drop=True)  


class FineTunedBertRetriever:
    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        self.document_embeddings_path = f"../out/embeddings/document_embeddings_{model_name}.json"
        self.df_embeddings = self._load_document_embeddings(
            path=self.document_embeddings_path
        )

    def retrieve(self, query: str, n=10) -> pd.DataFrame:
        query_embedding = self.model.encode(query)
        document_embeddings = np.vstack(self.df_embeddings["document_embedding"].values)

        top_indices, similarities = self.cosine_similarity(
            query_embedding, document_embeddings, n=10
        )
        result_df = self.df_embeddings.loc[top_indices, ["url"]].copy()
        result_df["cosine_similarity"] = similarities
        return result_df.reset_index(drop=True)  

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


if __name__ == "__main__":
    # example code
    retriever = BioASQEmbeddingRetriever()
    result_df = retriever.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10)
    print(result_df)

    retriever = FineTunedBertRetriever()
    result_df = retriever.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10)
    print(result_df)
    