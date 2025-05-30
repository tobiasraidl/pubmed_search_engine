# nlp_ir/src/sentence_embedder.py
from sentence_transformers import SentenceTransformer

"""
This class uses the Sentence-Transformers library to encode sentences into embeddings. 
It initializes a pre-trained model and provides methods to encode a list of texts or a single text. 
The embeddings are returned as NumPy arrays.
"""

class SentenceEmbedder:
    def __init__(self, model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, convert_to_numpy=True, **kwargs)

    def encode_single(self, text):
        return self.encode([text])[0]
