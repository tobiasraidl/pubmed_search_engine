import numpy as np
import json
import pandas as pd
from embedding_model import EmbeddingModel


class IRRertriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.document_embeddings_path = "../data/document_embeddings.json"
        self.df_embeddings = self._load_document_embeddings(
            path=self.document_embeddings_path
        )

    def _transform_query(self, query: str) -> np.array:
        """
        Transforms query to 200-dimensional embedding
        """
        return self.embedding_model.transform_query(query=query)

    def _load_document_embeddings(self, path) -> pd.DataFrame:
        """
        Loads url and document embeddings from JSON and returns dataframe.
        """
        with open(path, "r") as file:
            document_embeddings = json.load(file)
        return pd.DataFrame(document_embeddings)

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

    def retrieve(self, query: str, n=10):
        query_embedding = self._transform_query(query)
        document_embeddings = np.vstack(self.df_embeddings["document_embedding"].values)

        top_indices, similarities = self.cosine_similarity(
            query_embedding, document_embeddings, n=10
        )
        result_df = self.df_embeddings.loc[top_indices, ["url"]].copy()
        result_df["cosine_similarity"] = similarities
        return result_df.reset_index(
            drop=True
        )  # self.df_embeddings.iloc[top_indices].reset_index(drop=True)


if __name__ == "__main__":
    retriever = IRRertriever()

    result_df = retriever.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10)

    print(result_df)
