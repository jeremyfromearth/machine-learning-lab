import re
import numpy as np
import pandas as pd

class TermFreqInverseDocFreq:
    def __init__(self):
        self.term_frequency = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.word_count_over_iterations = []
        self.tfidf = None

    def create(self, corpus, text_column, normalize = True):
        term_freq = {}
        for index, row in corpus.iterrows():
            if index in term_freq:
                terms = term_freq[index]
            else:
                terms = term_freq[index] = {}

            s = str(row[text_column]).lower()
            word_list = re.findall(r'\b[^\W\d_]+\b', s) 

            for word in word_list:
                if word in terms:
                    terms[word] += 1
                else:
                    terms[word] = 1
            self.word_count_over_iterations.append(len(terms))
        max = 0
        for k, v in term_freq.items():
            if len(v) > max:
                max = len(v)

        print('Term frequency dict created')
        # create a DataFrame of the term frequencies
        self.term_frequency = pd.DataFrame(term_freq)    
        print('Term frequency DataFrame created')
        # fill missing values with zero
        self.term_frequency = self.term_frequency.fillna(0)               
        print('Term frequency na filled')
        # normalize each row so that it's magnitude == 1.0
        if normalize:
            self.term_frequency = self.term_frequency.apply(lambda x : x / np.sqrt(np.sum(np.power(x.values, 2)))) 
            print('Term frequency normalized')
        # conver the term frequency to a Series representing document frequency
        self.document_frequency = self.term_frequency.astype(bool).sum(axis=1)
        print('Document frequency DataFrame created')
        # create the inverse document frequency dataframe
        self.inverse_document_frequency = \
            self.document_frequency.apply(lambda x : np.log(self.term_frequency.shape[1] / (1 + x))) 
        print('Inverse document frequency DataFrame created')
        # create the tfidf
        self.tfidf = \
            pd.DataFrame(
                self.term_frequency.values.T * self.inverse_document_frequency.values, 
                index=self.term_frequency.columns, columns=self.inverse_document_frequency.index)
        
        print('TFIDF DataFrame created')
