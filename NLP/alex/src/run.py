import os
import torch
from bioasq_embedding_model import BioASQEmbeddingModel
from sentence_transformers import SentenceTransformer
from fine_tune_bi_encoder import fine_tune
from gen_train_triples import gen_train_triples
from create_document_embeddings import create_embeddings
from retriever import BioASQEmbeddingRetriever, FineTunedBertRetriever
from evaluate import evaluate_and_store_results

BERT_NAME = "all-MiniLM-L6-v2"
BERT_PATH = "sentence-transformers/all-MiniLM-L6-v2"


def initilaize_models() -> tuple[BioASQEmbeddingModel, SentenceTransformer]:
    bioasq_model = BioASQEmbeddingModel()
    bert_model = SentenceTransformer(BERT_PATH)
    return bioasq_model, bert_model


def load_bert_model(model_path) -> SentenceTransformer:
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_path, device=device)


def get_fine_tuned_bert_model(bert_model) -> SentenceTransformer:
    fine_tuned_model_path = f"../out/models/{BERT_NAME}-fine-tuned"

    if os.path.exists(fine_tuned_model_path):
        print("Model exists. Returning model.")
        return load_bert_model(fine_tuned_model_path), fine_tuned_model_path
    else:
        print("Model doesn't exist.")

        if os.path.exists("../out/triples.jsonl"):
            print("Start fine tuning")
            model_path = fine_tune(model=bert_model, model_name=BERT_NAME, model_path=fine_tuned_model_path)
            print(f"Model Path: {model_path}")
            return load_bert_model(model_path), model_path
        else:
            print("Training triplets do not exist - start generating triplets.")
            gen_train_triples(negatives_per_positives=3)
            print("Start fine tuning")
            model_path = fine_tune(model=bert_model, model_name=BERT_NAME, model_path=fine_tuned_model_path)
            print(f"Model Path: {model_path}")
            return load_bert_model(model_path), model_path




def main() -> None:
    # load models
    bioasq_model, bert_model = initilaize_models()
    fine_tuned_bert_model, fine_tuned_model_path = get_fine_tuned_bert_model(bert_model)
    
    models = {
        "BioASQEmbeddingModel" : bioasq_model,
        "all-MiniLM-L6-v2" : bert_model,
        "all-MiniLM-L6-v2-fine-tuned" : fine_tuned_bert_model
    }

    # create document embeddings using different models
    create_embeddings(models)

    # initilaize retriever
    bio_asq_retriever = BioASQEmbeddingRetriever()
    bert_retriever = FineTunedBertRetriever(BERT_PATH)
    fine_tuned_name = BERT_NAME + "-fine-tuned"
    bert_fine_tuned_retriever = FineTunedBertRetriever(model_path=fine_tuned_model_path, model_name=fine_tuned_name)
    
    # evaluate performance on test set
    retrievers = {
        "BioASQEmbeddingModel" : bio_asq_retriever,
        "all-MiniLM-L6-v2" : bert_retriever,
        "all-MiniLM-L6-v2-fine-tuned" : bert_fine_tuned_retriever
    }
    evaluate_and_store_results(retrievers)

    print("Done.")

if __name__ == "__main__":
    main()