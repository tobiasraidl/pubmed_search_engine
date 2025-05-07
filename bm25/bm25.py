from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import json
from tqdm import tqdm

class BM25_retriever():
    def __init__(self, tokenized_documents_path, documents_path):
        self.model, self.tokenized_documents, self.documents = self.build_retriever(tokenized_documents_path, documents_path)

    def build_retriever(self, tokenized_documents_path, documents_path):
        with open(tokenized_documents_path, "r", encoding="utf-8") as f:
            tokenized_documents = json.load(f)

        with open(documents_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        bm25 = BM25Okapi(tokenized_documents)
        return bm25, tokenized_documents, documents
    
    def query(self, query, top_n=10):
        doc_urls = [doc['url'] for doc in self.documents]
        tokenized_query = word_tokenize(query.lower())
        scores = self.model.get_scores(tokenized_query)
        # top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        top_n_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_n_indices = top_n_indices[np.argsort(scores[top_n_indices])[::-1]]

        return [doc_urls[i] for i in top_n_indices]
    
    def run_all_queries(self, questions_path, out_path, top_n=10):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            questions["questions"] = [{"body": q["body"], "id": q["id"]} for q in questions["questions"]]
        
        for i, question in tqdm(enumerate(questions["questions"]), desc="Querying"):
            query = question["body"]
            questions["questions"][i]["documents"] = self.query(query, top_n)
            
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(questions, f)
            
    def evaluate(self, questions_path):
        """This function requires the questions pathto include the relevant labeled documents under the key "documents" """
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        precisions, recalls, f1s = [], [], []
        
        for question in tqdm(questions["questions"], desc="Evaluating"):
            relevant = question["documents"]
            query = question["body"]
            predicted = self.query(query)
            
            precision = self._precision(relevant, predicted)
            recall = self._recall(relevant, predicted)
            f1 = self._f1_score(precision, recall)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        macro_f1 = sum(f1s) / len(f1s)
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
            
    @staticmethod
    def _precision(relevant, predicted):
        if not predicted:
            return 0.0
        return len(set(relevant) & set(predicted)) / len(predicted)
    
    @staticmethod
    def _recall(relevant, predicted):
        if not relevant:
            return 0.0
        return len(set(relevant) & set(predicted)) / len(relevant)

    @staticmethod
    def _f1_score(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    
if __name__ == "__main__":
    retriever = BM25_retriever(tokenized_documents_path="data/train/tokenized_documents.json", documents_path="data/train/documents.json")
    retriever.run_all_queries(questions_path="data/test/test_batch4.json", out_path="bm25/out/prediction_test13.json")
    # retriever.evaluate(questions_path="data/train/training13b.json")