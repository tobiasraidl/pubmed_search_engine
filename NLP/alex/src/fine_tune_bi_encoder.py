"""
This script fine-tunes a Siamese SentenceTransformer model (Bi-Encoder) using triplet loss.
It loads a list of query-positive-negative triples from a JSONL file,
converts them into training samples, and uses the TripletLoss objective
to train the model to bring semantically similar texts closer in embedding space,
and dissimilar ones farther apart.
"""

import logging
import os
import random
import json 
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sentence_transformers.evaluation import TripletEvaluator


random.seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

log_dir = '../out/logs'
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'fine-tuning.log')
logging.basicConfig(filename=log_file, level=logging.INFO)


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


def fine_tune(model, model_name, model_dir="../out/models", model_path=None) -> str:
    # training params
    batch_size = 32
    epochs = 5
    warmup_steps = 100
    model.tokenizer.model_max_length = 512
    model.tokenizer.truncation_side = 'right'
    
    if model_path is None:
        model_path = f"{model_dir}/{model_name}-fine-tuned-test"
    
    # load data
    triplets = load_pairs_from_triples("../out/triples.jsonl")[:10]

    # create train and validation splits
    val_size = int(len(triplets) * 0.1)
    val_samples = triplets[:val_size]
    train_samples = triplets[val_size:]
    print(f"Training size: {len(train_samples)}")
    print(f"Validation size: {len(val_samples)}")
    
    # create dataloader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size) # [:int((len(train_samples)*0.1))]
    
    # create evaluator from validation triplets
    evaluation_steps = int(len(train_samples) / batch_size) # once per epoch
    evaluator = TripletEvaluator.from_input_examples(
        val_samples, name="val-triplet-eval"
    )

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", model.device if hasattr(model, "device") else "unknown")

    # define loss
    train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE) # TripletDistanceMetric.COSINE, "cosine"

    # fine-tuning
    logging.info("Start fine-tuning")
    model.to(DEVICE)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5},
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        output_path=f"{model_dir}/{model_name}-fine-tuned",
        show_progress_bar=True
    )
    logging.info("Finished fine-tuning")

    # save training params
    training_params = {
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "max_length": model.tokenizer.model_max_length,
        "truncation_side": model.tokenizer.truncation_side
    }
    os.makedirs("../out/fine-tuning-params", exist_ok=True)
    with open("../out/fine-tuning-params/fine-tuning-params.json", "w", encoding="utf-8") as f:
        json.dump(training_params, f, indent=4, ensure_ascii=False)

    return model_path


if __name__ == "__main__":
    # paths, model names
    model_dir = "../out/models"
    models = {
        "all-MiniLM-L6-v2" : "sentence-transformers/all-MiniLM-L6-v2",
        # "biobert" : "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    }

    for model_name, model_path in tqdm(models.items(), desc="Fine-tuning models"):
        model = SentenceTransformer(model_path)
        fine_tune(model=model, model_name=model_name)
