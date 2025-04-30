from rank_bm25 import BM25Okapi
from preprocessing.read_training_data import get_session_key, retrieve_documents_from_api
import numpy as np

session_key = get_session_key()
query = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
documents = retrieve_documents_from_api(query=query, session_key=session_key)

corpus = []
pmids = []  # <-- Keep track of PMIDs here

for document in documents:
    document_contents = document["title"] + " " + document["abstract"]
    corpus.append(document_contents.split())
    pmids.append(document["id"])  # <-- Save the PMID

bm25 = BM25Okapi(corpus)
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)
ranked_doc_indices = np.argsort(scores)[::-1]

for idx in ranked_doc_indices:
    pmid = pmids[idx]  # <-- Use PMID instead of index
    print(f"PMID {pmid}: Score {scores[idx]}")
