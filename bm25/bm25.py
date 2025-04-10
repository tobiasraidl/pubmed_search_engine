from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('indexes/pubmed-bm25')
hits = searcher.search('hpv')

print(len(hits), "results")
for i, hit in enumerate(hits):
    print(f'{i+1}: {hit.docid} {hit.score}')
    if i > 10:
        break
