from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import json

class BM25_retriever():
    def __init__(self, tokenized_documents_path, documents_path):
        self.model, self.tokenized_documents, self.documents = self.build_retriever(tokenized_documents_path, documents_path)

    def build_retriever(self, tokenized_documents_path, documents_path):
        with open(tokenized_documents_path, "r") as f:
            tokenized_documents = json.load(f)

        with open(documents_path, "r") as f:
            documents = json.load(f)

        bm25 = BM25Okapi(tokenized_documents)
        return bm25, tokenized_documents, documents
    
    def query(self, query, top_n=10):
        doc_urls = [doc['url'] for doc in self.documents]
        tokenized_query = word_tokenize(query.lower())
        scores = self.model.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [doc_urls[i] for i in top_n_indices]
    
    def run_all_queries(self, questions_path, out_path, top_n=10):
        with open(questions_path, "r") as f:
            questions = json.load(f)
        
        for i, question in enumerate(questions["questions"]):
            query = question["body"]
            questions["questions"][i]["documents"] = self.query(query, top_n)
            
        with open(out_path, "w") as f:
            json.dump(questions, f)
        

    
if __name__ == "__main__":
    retriever = BM25_retriever("../data/train/tokenized_documents.json", "../data/train/documents.json")
    # results = retriever.query("Is Hirschsprung disease a mendelian or a multifactorial disorder?")
    # for url in results:
    #     print(url)
    
    retriever.run_all_queries("../data/test/questions.json", "../data/out/results.json")