import json
from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, ground_truth_path: str, predictions_path: str):
        self.ground_truth = self._load_json(ground_truth_path)
        self.predictions = self._load_json(predictions_path)
        self._validate_data()

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _validate_data(self):
        gt_questions = [q["body"] for q in self.ground_truth["questions"]]
        pred_questions = [q["body"] for q in self.predictions["questions"]]
        assert gt_questions == pred_questions, "Mismatch between ground truth and prediction questions."

    def _binarize(self, retrieved: List[str], relevant: Set[str], all_possible: Set[str]) -> List[int]:
        return [1 if doc in relevant else 0 for doc in retrieved]

    def evaluate(self) -> Dict[str, float]:
        precisions, recalls, f1s = [], [], []
        
        for gt_q, pred_q in zip(self.ground_truth["questions"], self.predictions["questions"]):
            relevant_docs = set(gt_q["documents"])
            retrieved_docs = pred_q["documents"]

            if not retrieved_docs and not relevant_docs:
                # Edge case: nothing to retrieve and nothing relevant — perfect match
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
                continue
            elif not retrieved_docs:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
                continue

            true_positives = len(set(retrieved_docs) & relevant_docs)
            precision = true_positives / len(retrieved_docs)
            recall = true_positives / len(relevant_docs) if relevant_docs else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s)
        }


if __name__ == "__main__":
    evaluator = Evaluator(ground_truth_path="data/train/training13b.json", predictions_path="bm25/out/prediction13b.json")
    print(evaluator.evaluate())