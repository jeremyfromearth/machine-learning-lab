import pandas as pd
from analytics import TermFreqInverseDocFreq

df = pd.read_csv('./data/fauna.csv.gz', compression='gzip')
df = df.set_index(keys=['page-id'])
tfidf = TermFreqInverseDocFreq()
tfidf.create(df[:3000], 'text', True)
print('complete')

