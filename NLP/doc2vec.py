# py -m pip install pandas 
import pandas as pd
from pathlib import Path
import json
import os
import smart_open
import gensim
import random


# set path to file
p = Path(r'C:\Users\kathr\OneDrive\uni\SS2025\VU AdvIR\Task1aDryRun_mid.json')

# read json
with p.open('r', encoding='utf-8') as f:
    data_dict = json.loads(f.read())

data = data_dict["documents"]

# create dataframe
df = pd.json_normalize(data)
print(df)

# save dataframe as csv
df.to_csv('Task1aDryRun.csv', encoding='utf-8', index=False)

# Set file names for train and test data
data_dir = Path(r'C:\Users\kathr\OneDrive\uni\SS2025\VU AdvIR') 
train_file = os.path.join(data_dir, 'Task1aDryRun_short.csv')   

# Define a Function to Read and Preprocess Text

tag_id_map = {}  # To map 'doc_0' -> original ID
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if i == 0:
                # skip header
                continue
              
            tokens = gensim.utils.simple_preprocess(line)
            
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                tag = f"doc_{i-1}"
                tag_id_map[tag] = df.iloc[i-1, 0]  # Store the original ID
                yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])

train_corpus = list(read_corpus(train_file))
  # print(train_corpus)


# Training the Model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=3, epochs=40)


# Build a vocabulary
model.build_vocab(train_corpus)

# test
# print(f"Word 'of' appeared {model.wv.get_vecattr('of', 'count')} times in the training corpus.")

# train the model on the corpus.
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# infer vector for piece of text
vector = model.infer_vector(['is', 'agomelatine', 'an', 'effective', 'drug'])
print(vector)

# Retrieving documents
  # testing id_23472096 (doc_2293)
test_corpus = ["is", "migalastat", "used", "for", "treatment", "of", "fabry", "disease"]
  # testing id 23581014 (doc_256)
test_corpus = ["which", "protein", "is", "the", "e3", "ubiquitin", "ligase", "that", "targets", "the", "tumor", "suppressor", "p53", "for", "proteasomal", "degradation"]
inferred_vector = model.infer_vector(test_corpus)
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
sims20 = model.dv.most_similar([inferred_vector], topn=20)
# print the most similar documents from the train corpus
print("Q: ", test_corpus)
print("Doc: ", df[df["pmid"]==tag_id_map[sims[0][0]]])
print(df[df["pmid"]==tag_id_map[sims[0][0]]]["title"])

for i, tu in enumerate(sims20):
  print("Doc: ", df[df["pmid"]==tag_id_map[sims[i][0]]])


## extra: reverse lookup of id/doc
id_tag_map = {v: k for k, v in tag_id_map.items()}
id_tag_map[23581014]
