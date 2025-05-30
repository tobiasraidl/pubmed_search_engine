# nlp_ir/src/build_document_embeddings.py
import json
import os
from sentence_embedder import SentenceEmbedder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

DOCS_PATH = "data/train/corpus.json"
OUTPUT_PATH = "nlp_ir/data/sbert_document_embeddings.json"

"""
This function is for building a document embedding using a pre-trained Sentence-BERT model.
It reads documents from a JSON file, encodes them into embeddings, and saves the results to a new JSON file. 
The embeddings are computed by concatenating the title and abstract of each document and passing them through the model. 
The output is a list of dictionaries containing the document URL and its corresponding embedding.
"""

def build_embeddings(model_path="nlp_ir/model/sbert-finetuned-bioasq"):
    embedder = SentenceEmbedder(model_path)
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    texts = [f"{d['title']} {d['abstract']}" for d in docs]
    vectors = embedder.encode(texts, show_progress_bar=True)

    results = []
    for doc, vec in zip(docs, vectors):
        results.append({"url": doc["url"], "document_embedding": vec.tolist()})

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    build_embeddings()
