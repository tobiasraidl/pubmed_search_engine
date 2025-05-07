from rank_bm25 import BM25Okapi
import string
import json
import torch
from sentence_transformers import CrossEncoder
import heapq
from tqdm import tqdm

# Preprocess the corpus (remove punctuation, tokenize, etc.)
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('[', '').replace(']', '')
    return text.lower().split()



class BM25Retriever:
    def __init__(self, documents_path):
        with open(documents_path, "r") as f:
            documents = json.load(f)
        self.corpus = [{"url": doc['url'], "text": f"{doc['title']} {doc['abstract']}"} for doc in documents]
        self.tokenized_corpus = [preprocess_text(doc['text']) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, n=10):
        tokenized_query = preprocess_text(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=n)
        return [{"url":doc["url"], "text": doc["text"]} for doc in results]
    
    def retrieve_all(self, queries, n=10):
        """
        Retrieves documents for a list of queries.

        Args:
            queries (list): _description_
            n (int): Number of documents returned for each query.

        Returns:
            (list): A list of dicts, where each dict contains the query and the top n documents for this query.
        """
        result = []
        for query in tqdm(queries, desc="Preprocessing queries", unit="query"):
            query_documents = self.retrieve(query, n=n)
            query_documents = [doc["url"] for doc in query_documents]
            result.append({"body": query, "documents": query_documents})
        return result
    
class Reranker:
    """
    A class to rerank documents using a CrossEncoder model.
    
    Attributes:
    
    Args:
        preranker_list (list): A list of documents to be reranked. Each document should be a dictionary with 'url' and 'text' keys.
        model_path (str): Path to the CrossEncoder model or the handle of a huggingface model. If not provided, the baseline model "cross-encoder/ms-marco-MiniLM-L-6-v2" will be used.
    """
    
    def __init__(self, model_path="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.bm25 = BM25Retriever("data/train/documents.json")
        self.model = CrossEncoder(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        
    def retrieve(self, query, n=10, preranker_n=100):
        bm25_result = self.bm25.retrieve(query, n=preranker_n)
        
        pairs = [[query, doc['text']] for doc in bm25_result]
        scores = self.model.predict(pairs)
        # reranked = sorted(zip([doc['url'] for doc in bm25_result], scores), key=lambda x: x[1], reverse=True)
        # truncated = reranked[:n]
        reranked_truncated = heapq.nlargest(
            n,
            zip([doc['url'] for doc in bm25_result], scores),
            key=lambda x: x[1]
        )
        return reranked_truncated
    
    def batch_retrieve(self, queries, n=10, preranker_n=100):
        """
        Retrieves documents for a list of queries.

        Args:
            queries (list): List of queries to retrieve documents for.
            n (int): Number of documents returned for each query.

        Returns:
            (list): A list of dicts, where each dict contains the query and the top n documents for this query.
        """
        results = []
        for query in tqdm(queries, desc="Preprocessing queries", unit="queries"):
            query_documents = self.retrieve(query, n=n, preranker_n=preranker_n)
            query_documents = [url for url, score in query_documents]
            results.append(query_documents)
        return results


if __name__ == "__main__":
    # BM25 retriever example
    bm25 = BM25Retriever("data/train/documents.json")
    bm25_results = bm25.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10)
    print("BM25 Results (URLs):")
    [print(doc["url"]) for doc in bm25_results]
    
    # BM25 + reranker example
    reranker = Reranker("reranker/out/model_1")
    reranked_docs = reranker.retrieve("Is Hirschsprung disease a mendelian or a multifactorial disorder?", n=10, preranker_n=1000)
    print("\nRe-ranked Documents (URLs with Scores):")
    for url, score in reranked_docs:
        print(f"URL: {url} | Score: {score}")
    