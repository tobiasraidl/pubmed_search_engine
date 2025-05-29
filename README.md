# s25_air
Contribution Chart
1. Preprocessing & Data Loading: Tobias, Alex G.
2. a traditional IR model (from the crash course): Tobias, Alex G.
3. neural NLP representation learning approach: Kathrin, Alex L., Saban
4. neural re-ranking model: Tobias, Alex G.
5. Evaluation: Tobias, Alex L.

### Setup
1. Install dependencies defined in requirements.txt
2. Download following nltk packages: `nltk.download('punkt')` and `nltk.download('stopwords')`
3. Generate test set specific corpus by running `python preprocessing/gen_corpus.py`. The original corpus is 38mio documents large. We reduce it to approximately 1000 times the number of queries in the test set by querying pubmed for 1000 documents per query. These are then accumulated to one large corpus that includes approximately 33% of documents that are labeled as relevant. The file `test/test`

### BM25 & Neural Reranker
The neural reranker model runs bm25 to preselect a specified number of documents before the reranker model itself is applied to select top-k documents per query. BM25 can also be used as a standalone. 

#### Finetune base model
Use `reranker/gen_train_triples` to generate training triples. Then finetune the baseline model in `reranker/train_reranker.py`.

#### Inference
Use `reranker/retriever.py` for inference tasks.

#### Evaluate
Both, BM25 and the neural reranker model are evaluated on 4 metrics (Precision@10, Recall@10, F1@10, MRR@10). This can be done by running the `reranker/evalute.py` file.
