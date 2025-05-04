"""
The pre-trained embedding models is capable to map to each word (key) and 200 dimension vector (value).
"""
import os
import numpy as np
from gensim.models import KeyedVectors


class EmbeddingModel:
    def __init__(self):
        self.path_to_words = "../data/types.txt" #"/Users/alexanderlorenz/Downloads/word2vecTools/types.txt"
        self.path_to_vectors = "../data/vectors.txt" # "/Users/alexanderlorenz/Downloads/word2vecTools/vectors.txt"
        self.model_path = "../model/bioasq_embedding_model.kv"

    def load_model(self, test_output=True, save=True):
        """
        Load pre-trained word embedding model
        """
        save = True

        if os.path.exists(self.model_path):
            kv = KeyedVectors.load(self.model_path)

            if test_output:
                print(kv["catel"].shape)

            return kv
        else:
            # load words
            with open(self.path_to_words, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f]
            # load vectors
            vectors = np.loadtxt(self.path_to_vectors)

            # init model, add vectors, and store model
            kv = KeyedVectors(vector_size=vectors.shape[1])
            kv.add_vectors(words, vectors)

            if save:
                kv.save(self.model_path)

            if test_output:
                print(kv["catel"].shape)

            return kv
        

if __name__ == "__main__":
    # load model
    print("Loading embedding model")
    kv = EmbeddingModel().load_model()

    print(kv["catel"])