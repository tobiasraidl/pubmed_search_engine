from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

print("Loading data...")
with open("reranker/triples.jsonl", "r") as f:
    triples = [json.loads(line) for line in f.readlines()]
    
train_examples = []
for triple in triples:
    train_examples.append(
        InputExample(texts=[triple["query"], triple["positive"]], label=1.0)
    )
    train_examples.append(
        InputExample(texts=[triple["query"], triple["negative"]], label=0.0)
    )
    
print("Creating dataloader...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

print("Training model...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="reranker/out/model_1",
)
print("Model trained and saved to reranker/out")