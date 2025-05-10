import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

"""
Section 1: Data Loading and Preprocessing
Load the BioASQ data from a JSON file and extracts the questions.
It then preprocesses the questions by tokenizing the text, removing stop words, and stemming the words using the NLTK library.
The preprocessed questions are stored in a list called preprocessed_questions.
"""
# Load BioASQ data
with open("data/train/training13b.json", "r") as f:
    data = json.load(f)

questions = data["questions"]

# Preprocess data
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

preprocessed_questions = [preprocess_text(question["body"]) for question in questions]


"""
Section 2: SBERT Model Creation
Create an SBERT model using the sentence-transformers library.
The model is initialized with a pre-trained MiniLM-L6-v2 model, which is a variant of the BERT model that is specifically designed for sentence embeddings.
"""
# Create SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

"""
Section 3: Fine-Tuning
Create a list of training examples, where each example consists of a question and a positive document.
The training examples are then used to create a data loader, which is used to feed the data to the model during training.
The model is fine-tuned using a cosine similarity loss function, which is designed to optimize the similarity between the question and positive document embeddings.
The model is trained for 1 epoch with a batch size of 16 and a warm-up period of 100 steps.
The trained model is saved to a file called sbert_model.
"""
# Fine-tune SBERT model
train_examples = []
for question in questions:
    positive_docs = question["documents"]
    for positive_doc in positive_docs:
        train_examples.append(InputExample(texts=[question["body"], positive_doc], label=1.0))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="nlp_ir/model/sbert_model-all-MiniLM-L6-v2",
    show_progress_bar=True,
)

"""
Section 4: Querying
Define a function called query_documents, which takes a question and a list of documents as input and returns a list of similarities between the question and each document.
The function uses the fine-tuned SBERT model to encode the question and documents, and then computes the cosine similarity between the question and each document embedding.
The similarities are returned as a list of floating-point numbers.
The function is then called with a sample question and a list of documents, and the similarities are printed to the console.
"""
# Use fine-tuned SBERT model for querying
def query_documents(question, documents):
    question_embedding = model.encode(question)
    document_embeddings = model.encode(documents)

    similarities = []
    for document_embedding in document_embeddings:
        similarity = torch.cosine_similarity(question_embedding, document_embedding)
        similarities.append(similarity.item())

    return similarities

question = "What is the treatment for diabetes?"
documents = ["The treatment for diabetes is insulin therapy.", "The treatment for diabetes is diet and exercise."]

similarities = query_documents(question, documents)
print(similarities)