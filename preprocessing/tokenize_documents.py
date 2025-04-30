from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import json
import os.path


def generate_corpus(path_raw_corpus, out_path):
    
    with open(path_raw_corpus, "r") as f:
        documents = json.load(f)

    corpus = [f"{doc['title']} {doc['abstract']}" for doc in documents]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    with open(out_path, "w") as f:
        json.dump(tokenized_corpus, f, indent=2)

if __name__ == "__main__":
    # nltk.download('punkt_tab')
    generate_corpus("data/train/documents.json", "data/train/tokenized_documents.json")