"""
This script fine-tunes a Siamese SentenceTransformer model (Bi-Encoder) using triplet loss.
It loads a list of query-positive-negative triples from a JSONL file,
converts them into training samples, and uses the TripletLoss objective
to train the model to bring semantically similar texts closer in embedding space,
and dissimilar ones farther apart.
"""


import random
import json 
import os
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm


random.seed(42)


def load_pairs_from_triples(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            pos = data["positive"]
            neg = data["negative"]
            samples.append(InputExample(texts=[query, pos, neg]))  # Positive pair

    return samples


if __name__ == "__main__":
    # paths, model names
    model_dir = "../out/models"
    models = {
        "all-MiniLM-L6-v2" : "sentence-transformers/all-MiniLM-L6-v2"
    }

    # load data
    train_samples = load_pairs_from_triples("../../../reranker/out/triples.jsonl")
    train_dataloader = DataLoader(train_samples[:50], shuffle=True, batch_size=32)

    for model_name in tqdm(models, desc="Fine-tuning models"):
        # init model
        model = SentenceTransformer(models[model_name]) # 'sentence-transformers/all-biomed-roberta-base', 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli'

        print("CUDA available:", torch.cuda.is_available())
        print("Using device:", model.device if hasattr(model, "device") else "unknown")

        # define loss config
        train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE) # TripletDistanceMetric.COSINE, "cosine"

        # fine-tune on triplets
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=3
        )

        # save model
        os.makedirs(model_dir, exist_ok=True)
        model.save(f"{model_dir}/{model_name}-fine-tuned")
