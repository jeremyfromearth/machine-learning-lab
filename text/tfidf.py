import re
import numpy as np
import pandas as pd

def tfidf(corpus):
    term_freq = {}
    for index, row in corpus.iterrows():
        if index in term_freq:
            terms = term_freq[index]
        else:
            terms = term_freq[index] = {}

        s = str(row['text']).lower()
        word_list = re.findall(r'\b[^\W\d_]+\b', s) 

        for word in word_list:
            if word in terms:
                terms[word] += 1
            else:
                terms[word] = 1

    tf = pd.DataFrame(term_freq)    # create a DataFrame of the term frequencies
    tf = tf.fillna(0)               # fill missing values with zero
    tf = tf.apply(lambda x : x / np.sqrt(np.sum(np.power(x.values, 2)))) # normalize each row so that it's magnitude == 1.0
    idf = tf.astype(bool).sum(axis=1).apply(lambda x : np.log(tf.shape[1] / (1 + x))) # inverse document frequency
    tfidf = pd.DataFrame(tf.values.T * idf.values, index=tf.columns, columns=idf.index)
    return tfidf
