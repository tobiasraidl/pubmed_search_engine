# Instructions

## Running BioASQEmbeddingModel
- Download pre-trained BioASQ embeddings here http://participants-area.bioasq.org/tools/BioASQword2vec/ 
- Move types.txt and vectors.txt to data folder
- The vector embeddings are stored separately (types.txt, vectors.txt), where the first word in types.txt corresponds to the first line in vectors.txt
- Execute create_document_embeddings.py, which maps the embeddings from BioASQ embeddings to each word of the corpus of a document (title + abstract), and computes the document embedding, and stores them
- Before running ir_retriever.py, maybe comment out some code in the main method

## Running FineTunedBertRetriever
- run fine_tune_bi_encoder.py to load, fine-tune, and locally the store the Bert model
- Execute create_document_embeddings.py, to precompute embeddings of the documents using the fine-tuned Bert model

# Todo:
- Write script to evaluate BioASQEmbeddingModel and FineTunedBertRetriever model performance