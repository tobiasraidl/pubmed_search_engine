import json
from retriever import BM25Retriever
from retriever import Reranker

    
def compute_metrics(predicted, ground_truth, k=10):
    assert len(predicted) == len(ground_truth), "Lists must have the same number of queries"

    total_prec, total_rec, total_f1, total_mrr = 0, 0, 0, 0
    num_queries = len(predicted)

    for preds, truths in zip(predicted, ground_truth):
        preds_k = preds[:k]
        truths_set = set(truths)

        hits = [doc for doc in preds_k if doc in truths_set]
        num_hits = len(hits)

        precision = num_hits / k
        recall = num_hits / len(truths) if truths else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Reciprocal Rank
        rr = 0
        for i, doc in enumerate(preds_k, start=1):
            if doc in truths_set:
                rr = 1 / i
                break

        total_prec += precision
        total_rec += recall
        total_f1 += f1
        total_mrr += rr

    return {
        "Precision@10": total_prec / num_queries,
        "Recall@10": total_rec / num_queries,
        "F1@10": total_f1 / num_queries,
        "MRR@10": total_mrr / num_queries
    }

    

def evaluate_bm25(gt_file, k=10):
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"]
    queries = [question["body"] for question in questions]
    relevant_documents = [question["documents"] for question in questions]
    
    retriever = BM25Retriever("data/train/documents.json")
    results = retriever.retrieve_all(queries, n=k)
    predicted_documents = [result["documents"] for result in results]
    
    return compute_metrics(predicted_documents, relevant_documents, k=k)


def evaluate_reranker(gt_file, model="cross-encoder/ms-marco-MiniLM-L-6-v2", k=10, preranker_n=100):
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"]
    queries = [question["body"] for question in questions]
    relevant_documents = [question["documents"] for question in questions]
    
    retriever = Reranker(model)
    predicted_documents = retriever.batch_retrieve(queries, n=k, preranker_n=preranker_n)
    
    return compute_metrics(predicted_documents, relevant_documents, k=k)

def export_bm25_predictions(questions_file, out_path, k=10):
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"]
    queries = [question["body"] for question in questions]
    retriever = BM25Retriever(questions_file)
    # batch_predicted_documents = retriever.batch_retrieve(queries, n=k, preranker_n=preranker_n)
    assert len(batch_predicted_documents) == len(questions), "Number of predicted documents must match number of questions"
    
    json_data = {"questions": []}
    for question, predicted_documents in zip(questions, batch_predicted_documents):
        json_data["questions"].append({"id": question["id"], "body": question["body"], "documents": predicted_documents})
    
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)

def export_reranker_predictions(questions_file, out_path, model="cross-encoder/ms-marco-MiniLM-L-6-v2", k=10, preranker_n=100):
    with open(questions_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"]
    queries = [question["body"] for question in questions]
    retriever = Reranker(model)
    batch_predicted_documents = retriever.batch_retrieve(queries, n=k, preranker_n=preranker_n)
    assert len(batch_predicted_documents) == len(questions), "Number of predicted documents must match number of questions"
    
    json_data = {"questions": []}
    for question, predicted_documents in zip(questions, batch_predicted_documents):
        json_data["questions"].append({"id": question["id"], "body": question["body"], "documents": predicted_documents})
    
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    # gt_file = "data/test/BioASQ-task13bPhaseB-testset3.json"
    # gt_file = "data/train/training13b.json"
    
    # bm25_results = evaluate_bm25(gt_file)
    # print(bm25_results)
    
    # finetuned_reranker_results = evaluate_reranker(gt_file, "reranker/out/models/", k=10, preranker_n=100)
    # print(finetuned_reranker_results)
    
    questions_file = "data/test/test_batch4.json"
    export_reranker_predictions(questions_file, "reranker/out/predictions/finetuned_reranker_prerank100.json", model="reranker/out/models/model_1", k=10, preranker_n=100)