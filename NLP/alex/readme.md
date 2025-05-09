# Instructions
- Download pre-trained BioASQ embeddings here http://participants-area.bioasq.org/tools/BioASQword2vec/ and move types.txt and vectors.txt to data folder
    - The vector embeddings are stored separately (types.txt, vectors.txt), in which the first word in types.txt corresponds to the first line in vectors.txt
- Execute create_document_embeddings.py, which maps the embeddings from BioASQ embeddings to each word of the corpus of a document (title + abstract), and computes the document embedding, and stores them

# Todo:
- write script that embeds query and calculates pair-wise cosine sim. with all document embeddings