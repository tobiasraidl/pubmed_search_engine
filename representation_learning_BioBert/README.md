# Representation Learning BioBert Experiment Descriptions
## representation_learning_BioBert/src/prepare_triplets.py
This prepares triplet data for training a model. It reads documents and questions from JSON files, constructs triplets of (query, positive, negative) samples, and saves them to a new JSON file. The triplets are built by selecting a random positive and negative document for each question, ensuring that the positive document is one of the documents associated with the question and the negative document is not.

## representation_learning_BioBert/src/train_sbert.py
This script trains a Sentence-BERT model using triplet loss. It loads triplet data from a JSON file, creates a DataLoader for batching, and trains the model for one epoch. The trained model is saved to a specified output path.

## representation_learning_BioBert/src/sentence_embbeder.py
The class `SentenceEmbedder` uses the Sentence-Transformers library to encode sentences into embeddings. It initializes a pre-trained model and provides methods to encode a list of texts or a single text. The embeddings are returned as NumPy arrays.

## representation_learning_BioBert/src/build_document_embeddings.py
This builds document embeddings using a pre-trained Sentence-BERT model. It reads documents from a JSON file, encodes them into embeddings, and saves the results to a new JSON file. The embeddings are computed by concatenating the title and abstract of each document and passing them through the model. The output is a list of dictionaries containing the document URL and its corresponding embedding.

## representation_learning_BioBert/src/sbert_retriever.py
The class `SBERTRetriever` retrieves documents based on a query using cosine similarity. It loads pre-computed document embeddings from a JSON file and uses a Sentence-BERT model to encode the query. The top N most similar documents are returned along with their similarity scores. The cosine similarity is computed between the query vector and the document matrix, and the results are sorted to find the most relevant documents.


# How to run the scripts
## 1. Prepare Triplet Data
```bash
python representation_learning_BioBert/src/prepare_triplets.py
```
Terminal output:
```
Building triplets: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5389/5389 [00:22<00:00, 235.13it/s]
```

## 2. Train SBERT Model
```bash
python representation_learning_BioBert/src/train_sbert.py
```
Terminal output:
```
{'train_runtime': 93.7594, 'train_samples_per_second': 16.318, 'train_steps_per_second': 1.067, 'train_loss': 0.4174595260620117, 'epoch': 10.0}                                                  
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:33<00:00,  1.07it/s]
```
## 3. Build Document Embeddings
```bash
python representation_learning_BioBert/src/build_document_embeddings.py
```
Terminal output:
```
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2642/2642 [08:11<00:00,  5.37it/s]
```
## 4. Retrieve Documents
```bash
python representation_learning_BioBert/src/sbert_retriever.py
```
Terminal output:
```
                                           url     score
0  http://www.ncbi.nlm.nih.gov/pubmed/37496148  0.706919
1  http://www.ncbi.nlm.nih.gov/pubmed/37496148  0.706919
2    http://www.ncbi.nlm.nih.gov/pubmed/880742  0.678129
3  http://www.ncbi.nlm.nih.gov/pubmed/26036949  0.671953
4  http://www.ncbi.nlm.nih.gov/pubmed/12746391  0.667395
```