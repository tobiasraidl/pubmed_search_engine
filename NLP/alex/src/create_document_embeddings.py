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
from embedding_model import EmbeddingModel


PATH_TO_DOCUMENTS = "../../../data/train/documents.json"


with open(PATH_TO_DOCUMENTS, "r") as file:
    data = json.load(file)


if __name__ == "__main__":
    # load model
    print("Loading embedding model")
    model = EmbeddingModel()

    # embed title and abstract
    print("Embed title and abstract")
    document_embeddings = []

    for i, document in enumerate(tqdm(data, desc="Embedding documents")):
        document_corpus = document["title"] + document["abstract"]
        document_embedding = model.transform_query(document_corpus)
        document_embeddings.append(
            {
                "url": str(document["url"]),
                "document_embedding": document_embedding.tolist(),
            }
        )

    # store document embeddings
    print("Store embeddings to /data")
    os.makedirs("../data/", exist_ok=True)
    with open("../data/document_embeddings.json", "w") as f:
        json.dump(document_embeddings, f)
