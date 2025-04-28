from rank_bm25 import BM25Okapi
from preprocessing.read_training_data import get_session_key, retrieve_documents_from_api
import numpy as np

session_key = get_session_key()
query = "nitric oxide synthase"
documents = retrieve_documents_from_api(query=query, session_key=session_key)

corpus = []
for document in documents:
    document_contents = document["title"] + " " + document["abstract"]
    corpus.append(document_contents.split())

bm25 = BM25Okapi(corpus)
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)
ranked_doc_indices = np.argsort(scores)[::-1]
for idx in ranked_doc_indices:
    print(f"Doc {idx}: Score {scores[idx]}")
