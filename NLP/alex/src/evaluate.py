import json
import os
from retriever import BioASQEmbeddingRetriever, FineTunedBertRetriever

    
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


def evaluate_model_performance(gt_file, retriever, k=10):
    with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    questions = gt_data["questions"]
    queries = [question["body"] for question in questions]
    relevant_documents = [question["documents"] for question in questions]

    #retriever = model_class()
    results = retriever.batch_retrieve(queries, n=k)
    predicted_documents = [result["documents"] for result in results]

    return compute_metrics(predicted_documents, relevant_documents, k=k)

if __name__ == "__main__":
    # path to labeled test data (ground truth)
    gt_file = "../../../data/test/test_batch3_with_gt.json" #"data/test/test_batch3_with_gt.json"
    os.makedirs("../out/results/", exist_ok=True)

    # evaluate BioASQEmbeddingRetriever
    results_bioasq = evaluate_model_performance(gt_file=gt_file, retriever=BioASQEmbeddingRetriever())
    with open("../out/results/bioasq_result.json", "w", encoding="utf-8") as f:
        json.dump(results_bioasq, f, indent=4, ensure_ascii=False)

    # evaluate FineTunedBertRetriever
    results_fine_tuned_model = evaluate_model_performance(gt_file=gt_file, retriever=FineTunedBertRetriever())
    with open("../out/results/fine_tuned_model_result.json", "w", encoding="utf-8") as f:
        json.dump(results_fine_tuned_model, f, indent=4, ensure_ascii=False)