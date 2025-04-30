from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import json
import os.path

# nltk.download('punkt_tab')
with open("data/train/corpus.json", "r") as f:
    documents = json.load(f)

if not os.path.exists("data/train/tokenized_corpus.json"):
    corpus = [f"{doc['title']} {doc['abstract']}" for doc in documents]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    with open("data/train/tokenized_corpus.json", "w") as f:
        json.dump(tokenized_corpus, f)
else:
    with open("data/train/tokenized_corpus.json", "r") as f:
        tokenized_corpus = json.load(f)

bm25 = BM25Okapi(tokenized_corpus)
doc_urls = [doc['url'] for doc in documents]

def query_bm25(query, top_n=3):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [doc_urls[i] for i in top_n_indices]

results = query_bm25("Concizumab is used for which diseases", top_n=10)
print("Top matching URLs:")
for url in results:
    print(url)
