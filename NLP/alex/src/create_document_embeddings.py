"""
This scripts embeds the title + abtract into a 200-dimensional
"document embedding". The pre-trained BioASQ vector embeddings are
stored separately: the first word in types.txt corresponds to the first line in vectors.txt.

This model is applied to map each word in a corpus to its corresponding
pre-trained vector, if available. Afterward, the document embedding is computed
by averaging each dimension across all avaiable word vectors in the document.
"""

import json
import re
import os
import numpy as np
from tqdm import tqdm
from bioasq_embedding_model import BioASQEmbeddingModel
from sentence_transformers import SentenceTransformer


PATH_TO_DOCUMENTS = "../../../data/train/documents.json"


with open(PATH_TO_DOCUMENTS, "r") as file:
    data = json.load(file)


if __name__ == "__main__":
    # load model
    print("Loading embedding model")
    
    models = {
        "BioASQEmbeddingModel" : BioASQEmbeddingModel(),
        "all-MiniLM-L6-v2" : SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    }

    # embed title and abstract
    print("Embed title and abstract")
    
    for model_name in models:
        document_embeddings = []
        model = models[model_name]
        for i, document in enumerate(tqdm(data, desc="Embedding documents")):
            document_corpus = document["title"] + document["abstract"]
            document_embedding = model.encode(document_corpus)
            document_embeddings.append(
                {
                    "url": str(document["url"]),
                    "document_embedding": document_embedding.tolist(),
                }
            )

        # store document embeddings
        print("Store embeddings to /data")
        os.makedirs("../out/embeddings", exist_ok=True)
        with open(f"../out/embeddings/document_embeddings_{model_name}.json", "w") as f:
            json.dump(document_embeddings, f)
