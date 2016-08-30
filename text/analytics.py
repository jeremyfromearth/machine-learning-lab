import re
import numpy as np
import pandas as pd

class TermFreqInverseDocFreq:
    def __init__(self):
        self.document_ids = None
        self.unique_words = None
        self.term_frequency = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.word_count_over_iterations = []
        self.table = None

    def create(self, corpus, text_column, normalize = True):
        term_freq = {}
        unique_words = set()
        text = {}
        for index, row in corpus.iterrows():
            s = str(row[text_column]).lower()
            word_list = re.findall(r'\b[^\W\d_]+\b', s) 
            text[index] = word_list

            for word in word_list:
                unique_words.add(word)

            self.word_count_over_iterations.append(len(unique_words))

        print('unique_words created')

        self.unique_words = sorted(list(unique_words))
        for index, row in corpus.iterrows():
            terms = np.zeros(len(self.unique_words))
            for word in text[index]:
                terms[self.unique_words.index(word)] += 1
            term_freq[index] = terms
            print('term_freq created for document', index)

        print(len(term_freq), len(term_freq[38082824]))
        print(unique_words)
        
        '''
        print('Term frequency dict created')
        self.document_ids = list(term_freq.keys())

        tf = []
        self.all_words = sorted(list(unique_words))
        for id in self.document_ids:
            counts = []
            for word in self.all_words:
                count = term_freq[id][word] if word in term_freq[id] else 0
            tf.append(counts)
            
        print(len(tf), len(tf[0]))
        '''

        '''
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
        self.table = \
            pd.DataFrame(
                self.term_frequency.values.T * self.inverse_document_frequency.values, 
                index=self.term_frequency.columns, columns=self.inverse_document_frequency.index)
        print('TFIDF DataFrame created')
        '''
