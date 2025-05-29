import json
import time
import re
from pathlib import Path
from Bio import Entrez
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

# === SETTINGS ===
EMAIL      = "apanci2000@gmail.com"           # your NCBI-registered email
API_KEY    = "a417cd398ae9ba5622989a1f8ef153750f08"               # your NCBI API key
Entrez.email   = EMAIL
Entrez.api_key = API_KEY

RETMX   = 1000     # number of documents to fetch per question
TOP_K   = 10       # number of documents to keep after BM25 re-ranking

# Input/Output paths
INPUT_FILE  = Path("data/test/test_batch3_with_gt.json")
OUTPUT_FILE = Path("data/out/test_batch3_out.json")

# === PREPROCESSING TOOLS ===
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def preprocess(text: str):
    """Lowercase, remove non-alphanumerics, tokenize, remove stopwords, stem."""
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    tokens = [stemmer.stem(tok)
              for tok in word_tokenize(text)
              if tok not in stop_words and len(tok) >= 2]
    return tokens

# === ESEARCH FUNCTION ===
def esearch_pmids(query: str, retmax: int = RETMX):
    """Fetch PMIDs for a given query string from PubMed."""
    # build simple term: Title/Abstract + MeSH
    phrase, tokens = query.lower(), query.lower().split()
    parts = [f'"{phrase}"[Title/Abstract]', f'"{phrase}"[MeSH Terms]']
    for t in tokens:
        parts += [f'{t}[Title/Abstract]', f'{t}[MeSH Terms]']
    term = f"({' OR '.join(parts)}) AND hasabstract[text]"

    for attempt in range(3):
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=term,
                retmax=retmax,
                sort="relevance",
                retmode="xml"
            )
            result = Entrez.read(handle)
            time.sleep(0.1)  # respect rate limit
            return result.get('IdList', [])
        except Exception:
            time.sleep(2 ** attempt)
    return []

# === MAIN PIPELINE ===

def main():
    data = json.loads(INPUT_FILE.read_text())['questions']

    output = []
    all_docs = []  # ← Store all (pmid, text) pairs here for entire corpus
    cnt = 1
    for item in data:
        print(f"On step {cnt} out of {len(data)} PMIDs")
        qid   = item['id']
        qtype = item.get('type')
        body  = item['body']

        # 1) fetch
        pmids = esearch_pmids(body, RETMX)

        # 2) fetch abstracts in batches
        docs = []
        batch_size = 1000
        for bidx in range(0, len(pmids), batch_size):
            batch = pmids[bidx:bidx+batch_size]
            print(f"    Fetching batch {bidx//batch_size+1} with {len(batch)} PMIDs...")
            handle = Entrez.efetch(db="pubmed",
                                   id=','.join(batch),
                                   retmode="xml")
            records = Entrez.read(handle)
            time.sleep(0.34)  # ~3 requests/sec
            for rec in records.get('PubmedArticle', []):
                pmid = rec['MedlineCitation']['PMID']
                art  = rec['MedlineCitation']['Article']
                title = art.get('ArticleTitle', '')
                abst = ''
                if art.get('Abstract'):
                    abst = ' '.join(art['Abstract']['AbstractText'])
                docs.append((pmid, f"{title} {abst}"))
        print(f"  Fetched {len(docs)} documents (title+abstract)")

        all_docs.extend(docs)  # ← Add to global corpus
        cnt += 1

    # Preprocess and save the entire corpus
    corpus = [preprocess(text) for _, text in all_docs]
    with open("data/train/corpus.json", "w") as f:
        json.dump(corpus, f)
    print("Saved full corpus to data/train/corpus.json")


if __name__ == '__main__':
    main()
