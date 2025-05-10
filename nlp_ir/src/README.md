# Script Descriptions
## nlp_ir/src/prepare_triplets.py
This prepares triplet data for training a model. It reads documents and questions from JSON files, constructs triplets of (query, positive, negative) samples, and saves them to a new JSON file. The triplets are built by selecting a random positive and negative document for each question, ensuring that the positive document is one of the documents associated with the question and the negative document is not.

## nlp_ir/src/train_sbert.py
This script trains a Sentence-BERT model using triplet loss. It loads triplet data from a JSON file, creates a DataLoader for batching, and trains the model for one epoch. The trained model is saved to a specified output path.

## nlp_ir/src/sentence_embbeder.py
The class `SentenceEmbedder` uses the Sentence-Transformers library to encode sentences into embeddings. It initializes a pre-trained model and provides methods to encode a list of texts or a single text. The embeddings are returned as NumPy arrays.

## nlp_ir/src/build_document_embeddings.py
This builds document embeddings using a pre-trained Sentence-BERT model. It reads documents from a JSON file, encodes them into embeddings, and saves the results to a new JSON file. The embeddings are computed by concatenating the title and abstract of each document and passing them through the model. The output is a list of dictionaries containing the document URL and its corresponding embedding.

## nlp_ir/src/sbert_retriever.py
The class `SBERTRetriever` retrieves documents based on a query using cosine similarity. It loads pre-computed document embeddings from a JSON file and uses a Sentence-BERT model to encode the query. The top N most similar documents are returned along with their similarity scores. The cosine similarity is computed between the query vector and the document matrix, and the results are sorted to find the most relevant documents.


# How to run the scripts
## 1. Prepare Triplet Data
```bash
python nlp_ir/src/prepare_triplets.py
```
Terminal output:
```
Building triplets: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5389/5389 [00:15<00:00, 348.85it/s]
```

## 2. Train SBERT Model
```bash
python nlp_ir/src/train_sbert.py
```
Terminal output:
```
{'train_runtime': 296.1853, 'train_samples_per_second': 18.195, 'train_steps_per_second': 1.138, 'train_loss': 0.6457651743903005, 'epoch': 1.0}                                                  
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 337/337 [04:56<00:00,  1.14it/s]
```
## 3. Build Document Embeddings
```bash
python nlp_ir/src/build_document_embeddings.py
```
Terminal output:
```
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1551/1551 [04:44<00:00,  5.45it/s]
```
## 4. Retrieve Documents
```bash
python nlp_ir/src/sbert_retriever.py
```
Terminal output:
```
                                           url     score
0  http://www.ncbi.nlm.nih.gov/pubmed/23001136  0.954175
1  http://www.ncbi.nlm.nih.gov/pubmed/12239580  0.931006
2  http://www.ncbi.nlm.nih.gov/pubmed/15617541  0.924503
3  http://www.ncbi.nlm.nih.gov/pubmed/17965226  0.918449
4  http://www.ncbi.nlm.nih.gov/pubmed/11106284  0.915042
```