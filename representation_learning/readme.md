# Instructions

- 1. Install dependencies

- 2. Download pre-trained BioASQ embeddings [here](http://participants-area.bioasq.org/tools/BioASQword2vec/)
    - Unpack zip and move `types.txt` and `vectors.txt` to data folder
    - (First word in `types.txt` corresponds to the first line in `vectors.txt`.)

- 3. Run `main.py` which handles the whole pipeline:
    
    1. Loading and initializing models  
    2. Fine-tuning models if needed  
    3. Creating document embeddings  
    4. Initializing retrievers  
    5. Evaluating retriever performance
