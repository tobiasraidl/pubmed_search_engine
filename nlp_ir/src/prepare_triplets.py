# nlp_ir/src/prepare_triplets.py
import json
import random
from tqdm import tqdm

DOCS_PATH = "data/train/corpus.json"
QUESTIONS_PATH = "data/train/training13b_clean.json"
OUTPUT_PATH = "nlp_ir/data/sbert_triplets.json"

"""
This is for preparing triplet data for training a model. 
It reads documents and questions from JSON files, constructs 
triplets of (query, positive, negative) samples, and saves them to a new JSON file.
The triplets are built by selecting a random positive and negative document for each question, 
ensuring that the positive document is one of the documents associated with the question and the negative document is not.
"""

def build_triplets():
    with open(DOCS_PATH) as f:
        docs = json.load(f)
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    url_to_doc = {doc["url"]: f"{doc['title']} {doc['abstract']}" for doc in docs}
    all_urls = list(url_to_doc.keys())
    triplets = []

    for q in tqdm(questions, desc="Building triplets"):
        query = q["body"]
        positives = [url for url in q["documents"] if url in url_to_doc]
        negatives = list(set(all_urls) - set(positives))
        if not positives or not negatives:
            continue
        triplets.append({
            "query": query,
            "positive": url_to_doc[random.choice(positives)],
            "negative": url_to_doc[random.choice(negatives)]
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(triplets, f, indent=2)

if __name__ == "__main__":
    build_triplets()
