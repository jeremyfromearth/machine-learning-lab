import sys
import pandas as pd
from analytics import TermFreqInverseDocFreq

df = pd.read_csv('./data/fauna.csv.gz', compression='gzip')
df = df.set_index(keys=['page-id'])

tfidf = TermFreqInverseDocFreq()
tfidf.create(df, 'text', True)

term = 'antelope'
doc_id = 1369072
doc_index = tfidf.doc_id_to_index[doc_id]
term_index = tfidf.terms[term]

print('Document Id:', doc_id, 'Term:', term) 
print('Document Index', doc_index, 'Term Index', term_index)
print('Local Frequency', tfidf.get_local_freq(doc_id, term))
print('Global Frequency: ', tfidf.get_doc_freq(term))
print('Global Rarity Score:', tfidf.get_global_rarity(term))
print('TFIDF Score for document', tfidf[doc_id][0, term_index])
print('TFIDF.shape', tfidf[doc_id].shape)
