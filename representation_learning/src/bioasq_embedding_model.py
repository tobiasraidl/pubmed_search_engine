"""
The pre-trained embedding models is capable to map to each word (key) and 200 dimension vector (value).
"""

import os
import re
import numpy as np
from gensim.models import KeyedVectors


class BioASQEmbeddingModel:
    def __init__(self):
        self.path_to_words = "../data/types.txt"  # "/Users/alexanderlorenz/Downloads/word2vecTools/types.txt"
        self.path_to_vectors = "../data/vectors.txt"  # "/Users/alexanderlorenz/Downloads/word2vecTools/vectors.txt"
        self.model_dir = "../out/models/"
        self.model_path = "../out/models/bioasq_embedding_model.kv"
        self.kv = self._load_model(verbose=True, save=True)

    def _load_model(self, verbose=True, save=True):
        """
        Load pre-trained word embedding model
        """
        save = True

        if os.path.exists(self.model_path):
            kv = KeyedVectors.load(self.model_path)

            if verbose:
                print(kv["catel"].shape)

            return kv
        else:
            print(
                "Loading model for the first time. Initialize model, store, and return it."
            )
            # load words
            with open(self.path_to_words, "r", encoding="utf-8") as f:
                words = [line.strip() for line in f]
            # load vectors
            vectors = np.loadtxt(self.path_to_vectors)

            # init model, add vectors, and store model
            kv = KeyedVectors(vector_size=vectors.shape[1])
            kv.add_vectors(words, vectors)

            # store kv model
            if save and not os.path.exists(self.model_path):
                os.makedirs(self.model_dir, exist_ok=True)
                kv.save(self.model_path)

            if verbose:
                print(kv["catel"].shape)

            return kv

    def bioclean(self, t):
        """Cleans strings just as the authors of BioASQ Word2Vec embeddings"""
        return " ".join(
            re.sub(
                "[.,?;*!%^&_+():-\[\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .strip()
                .lower(),
            ).split()
        )

    def encode(self, query: str) -> np.array:
        """
        Transforms query to 200-dimensional query embedding.
        """
        vectors = [
            self.kv[token] for token in self.bioclean(query).split() if token in self.kv
        ]
        document_embedding = np.mean(vectors, axis=0)

        return document_embedding


if __name__ == "__main__":
    # load model
    print("Loading embedding model")
    model = BioASQEmbeddingModel()

    # t = model.kv
    # print(t["catel"])
