import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
		
        top_doc_indices = np.argsort(W[:,topic_idx] )[::-1][0:no_top_documents]
        #for doc_index in top_doc_indices:
            #print(documents[doc_index])

k_topics = 20
n_features = 10000
n_top_terms = 10

print('-----------------------------------')
print('NMF')
print('-----------------------------------')
dirname = os.path.dirname(__file__)
data = json.load(open(os.path.join(dirname, 'ai-articles.json')))
tfidf_vectorizer = TfidfVectorizer(max_features=n_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data)
tfidf_features = tfidf_vectorizer.get_feature_names()

nmf = NMF(
    n_components=k_topics, 
    random_state=0, 
    l1_ratio=0.1, 
    init='nndsvda'
).fit(tfidf)

# doc X topic
nmf_W = nmf.transform(tfidf)

# topic X term
nmf_H = nmf.components_

# get the top n docs for a topic
n = 10
topic = 16
top_n_docs_for_topic = np.argsort(nmf_W[:,topic])[::-1][0:n]
print('Top {} documents for topic {}'.format(n, topic))
print(top_n_docs_for_topic)

# what topics does this document belong to
print('Most relevant topics for top docs')
for i in range(0, n):
    doc_idx = top_n_docs_for_topic[i]
    topic_idxs = nmf_W[doc_idx].argsort()[::-1]
    print('{}:'.format(doc_idx), topic_idxs)

print('\n------------')
print(nmf_H.shape)

display_topics(nmf_H, nmf_W, tfidf_features, data, 10, 1)

def get_topics_for_terms(terms):
    for term in terms:
        term_idx = tfidf_features.index(term)
        topics = nmf_H[:,term_idx].argsort()[::-1]
        print('[{}]'.format(term_idx), '{}:'.format(term), '\nTopics:', topics)

get_topics_for_terms(['platform', 'ai', 'partnership', 'turing'])

# LDA model is not nearly as coherent as the NFM 
print('-----------------------------------')
print('LDA')
print('-----------------------------------')
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=50, stop_words='english')
tf = tf_vectorizer.fit_transform(data)
tf_features = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(
    n_components=k_topics, 
    max_iter=5, 
    doc_topic_prior=0.4,
    topic_word_prior=0.5,
    learning_method='online', 
    learning_offset=10.0, random_state=0
).fit(tf)


lda_W = nmf.transform(tfidf)
lda_H = lda.components_
display_topics(lda_H, lda_H, tf_features, data, 10, 1)
