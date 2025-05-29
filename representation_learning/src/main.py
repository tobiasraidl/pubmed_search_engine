"""
Main execution script for the biomedical document retrieval system.

This script coordinates the complete pipeline:
1. Loading and initializing models
2. Fine-tuning models if needed
3. Creating document embeddings
4. Initializing retrievers
5. Evaluating retriever performance

Models can be enabled/disabled using the MODEL_SELECTION configuration.
"""

import os
import logging
from typing import Dict, Tuple, Optional
import torch

from bioasq_embedding_model import BioASQEmbeddingModel
from sentence_transformers import SentenceTransformer
from fine_tune_bi_encoder import fine_tune
from gen_train_triples import gen_train_triples
from create_document_embeddings import create_embeddings
from retriever import BioASQEmbeddingRetriever, BertRetriever
from evaluate import evaluate_and_store_results


# Configuration constants
CONFIG = {
    "bert_name": "all-MiniLM-L6-v2",
    "bert_path": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": "../out",
    "models_dir": "../out/models",
    "triples_path": "../out/triples.jsonl",
    "negatives_per_positives": 1,
    "manual_fine_tuned_path": "../out/models/all-MiniLM-L6-v2-fine-tuned-final"
}

# Model selection configuration
# Set to True to include the model in the pipeline, False to exclude it
MODEL_SELECTION = {
    "bioasq": False,              # BioASQ embedding model
    "bert": False,                # Base BERT model
    "fine_tuned_bert": True      # Fine-tuned BERT model
}


def setup_logging() -> None:
    """Configure logging for the application."""
    log_dir = os.path.join(CONFIG["output_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )


def get_device() -> str:
    """Determine the appropriate device for model execution."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def initialize_bioasq_model() -> Optional[BioASQEmbeddingModel]:
    """Initialize the BioASQ embedding model if enabled."""
    if not MODEL_SELECTION["bioasq"]:
        logging.info("BioASQ model disabled, skipping initialization")
        return None
        
    logging.info("Initializing BioASQ embedding model")
    return BioASQEmbeddingModel()


def initialize_bert_model() -> Optional[SentenceTransformer]:
    """Initialize the base BERT model if enabled."""
    if not MODEL_SELECTION["bert"] and not MODEL_SELECTION["fine_tuned_bert"]:
        logging.info("BERT models disabled, skipping initialization")
        return None
        
    logging.info(f"Initializing base BERT model: {CONFIG['bert_path']}")
    return SentenceTransformer(CONFIG['bert_path'], device=get_device())


def load_fine_tuned_model(model_path: str) -> SentenceTransformer:
    """Load a fine-tuned model from disk."""
    logging.info(f"Loading fine-tuned model from: {model_path}")
    return SentenceTransformer(model_path, device=get_device())


def prepare_fine_tuned_model(base_model: SentenceTransformer) -> Tuple[Optional[SentenceTransformer], Optional[str]]:
    """
    Prepare a fine-tuned model, creating it if necessary.
    
    Args:
        base_model: The base model to fine-tune if needed
        
    Returns:
        Tuple of (fine-tuned model, model path) or (None, None) if disabled
    """
    if not MODEL_SELECTION["fine_tuned_bert"]:
        logging.info("Fine-tuned BERT model disabled, skipping preparation")
        return None, None
        
    fine_tuned_path = os.path.join(CONFIG["models_dir"], f"{CONFIG['bert_name']}-fine-tuned")
    
    # If model exists, load it
    if os.path.exists(fine_tuned_path):
        logging.info(f"Fine-tuned model already exists at: {fine_tuned_path}")
        return load_fine_tuned_model(fine_tuned_path), fine_tuned_path
    
    # If training data exists, fine-tune model
    if os.path.exists(CONFIG["triples_path"]):
        logging.info("Starting fine-tuning with existing training data")
    else:
        # Generate training data if it doesn't exist
        logging.info("Generating training triplets")
        gen_train_triples(negatives_per_positives=CONFIG["negatives_per_positives"])
    
    # Fine-tune the model
    logging.info("Starting fine-tuning process")
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    model_path = fine_tune(
        model=base_model,
        model_name=CONFIG["bert_name"],
        model_path=fine_tuned_path
    )
    
    return load_fine_tuned_model(model_path), model_path


def initialize_retrievers(fine_tuned_model_path: Optional[str]) -> Dict[str, object]:
    """
    Initialize all enabled retriever models.
    
    Args:
        fine_tuned_model_path: Path to the fine-tuned model, or None if disabled
        
    Returns:
        Dictionary mapping retriever names to retriever instances
    """
    logging.info("Initializing enabled retrievers")
    retrievers = {}
    
    if MODEL_SELECTION["bioasq"]:
        retrievers["BioASQEmbeddingModel"] = BioASQEmbeddingRetriever()
        
    if MODEL_SELECTION["bert"]:
        retrievers[CONFIG["bert_name"]] = BertRetriever(CONFIG["bert_path"])
        
    if MODEL_SELECTION["fine_tuned_bert"] and fine_tuned_model_path:
        retrievers[f"{CONFIG['bert_name']}-fine-tuned"] = BertRetriever(
            model_path=fine_tuned_model_path,
            model_name=f"{CONFIG['bert_name']}-fine-tuned"
        )
    
    return retrievers


def main() -> None:
    """Main execution function."""
    # Setup
    setup_logging()
    logging.info("Starting retrieval system pipeline")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Log enabled models
    logging.info(f"Enabled models: {[k for k, v in MODEL_SELECTION.items() if v]}")
    
    try:
        # Initialize models
        bioasq_model = initialize_bioasq_model()
        bert_model = initialize_bert_model()
        fine_tuned_model, fine_tuned_path = (None, None) if bert_model is None else prepare_fine_tuned_model(bert_model)
        
        # Create embeddings for enabled models
        models = {}
        if bioasq_model:
            models["BioASQEmbeddingModel"] = bioasq_model
        if MODEL_SELECTION["bert"] and bert_model:
            models[CONFIG["bert_name"]] = bert_model
        if fine_tuned_model:
            models[f"{CONFIG['bert_name']}-fine-tuned"] = fine_tuned_model
        
        if models:
            logging.info(f"Creating document embeddings for models: {list(models.keys())}")
            create_embeddings(models)
        else:
            logging.warning("No models enabled, skipping embedding creation")
        
        # Initialize retrievers and evaluate
        retrievers = initialize_retrievers(fine_tuned_path)
        if retrievers:
            logging.info(f"Evaluating retrievers: {list(retrievers.keys())}")
            evaluate_and_store_results(retrievers)
        else:
            logging.warning("No retrievers enabled, skipping evaluation")
        
        logging.info("Pipeline completed successfully")
    
    except Exception as e:
        logging.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise
        

if __name__ == "__main__":
    main()