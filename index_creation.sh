python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/out \
  --index indexes/pubmed-bm25 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw
