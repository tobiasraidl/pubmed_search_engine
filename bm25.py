from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('data/indexes/pubmed-bm25')
hits = searcher.search('hpv')

for i in range(10):
    print(f'{i+1}: {hits[i].docid} {hits[i].score}')
