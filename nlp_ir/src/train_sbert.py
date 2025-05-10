# nlp_ir/src/train_sbert.py
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

"""
This trains a Sentence-BERT model using triplet loss. 
It loads triplet data from a JSON file, creates a DataLoader for batching, and trains the model for one epoch. 
The trained model is saved to a specified output path.
"""

def load_triplets(path="nlp_ir/data/sbert_triplets.json"):
    with open(path) as f:
        triplets = json.load(f)
    return [InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in triplets]

def train():
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    model.to(DEVICE)

    train_data = load_triplets()
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    train_loss = losses.TripletLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path="nlp_ir/model/sbert-finetuned-bioasq"
    )

if __name__ == "__main__":
    train()
