import json
import random
from tqdm import tqdm


def gen_train_triples(negatives_per_positives=3):
    """generates training triples for any model that needs to be trained on triples.
    negatives_per_positives: number of negative samples per positive sample, normal is 3-10
    """
    with open("../../../data/train/training13b_clean.json", "r") as f:
        questions = json.load(f)
        
    with open("../../../data/train/documents.json", "r") as f:
        documents = json.load(f)
    
    triples = []
    for question in tqdm(questions, desc="Generating triples"):
        positive_docs_urls = question["documents"]
        for positive_doc_url in positive_docs_urls:
            # get the negative samples
            negative_docs = random.sample(documents, negatives_per_positives)
            for negative_doc in negative_docs:
                # make sure the negative doc is not in the positive docs
                if negative_doc["url"] not in positive_docs_urls:
                    # get the list entry in documents where url is positive_doc_url and get the title and abstract of this entry
                    positive_doc = next((doc for doc in documents if doc["url"] == positive_doc_url), None)
                    # check wheter the document exists in the corpus if not skip this entry
                    if positive_doc is None:
                        print(f"Document with url {positive_doc_url} not found in corpus")
                        continue
                    
                    triples.append({
                        "query": question["body"], 
                        "positive": f"{positive_doc['title']} {positive_doc['abstract']}", 
                        "negative": f"{negative_doc['title']} {negative_doc['abstract']}"
                        })
    
    with open("../out/triples.jsonl", "w") as f: 
        for item in triples:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    gen_train_triples(3)