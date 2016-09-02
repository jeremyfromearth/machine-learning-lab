import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

class TermFreqInverseDocFreq:
    def __init__(self):
        self.terms = {}
        self.term_frequency = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.word_count_over_iterations = []
        self.table = None

    def create(self, corpus, text_column, normalize = True):
        data = []
        indptr = [0]
        indices = []
        self.terms = {}
        page_id_to_index = {}
        document_frequency = {}
        for page_id, row in corpus.iterrows():
            s = str(row[text_column]).lower()
            word_list = re.findall(r'\b[^\W\d_]+\b', s)
            page_id_to_index.setdefault(page_id, len(page_id_to_index))
            for word in word_list:
                index = self.terms.setdefault(word, len(self.terms))
                indices.append(index)
                data.append(1)
                if word not in document_frequency:
                    document_frequency[word] = set()
                    document_frequency[word].add(page_id)
                else:
                    document_frequency[word].add(page_id)

            indptr.append(len(indices))
            self.word_count_over_iterations.append(len(self.terms))
       
        # create a sparse matrix representation of the term frequencies
        #
        # page-id, a, as, be, both
        # 2342342, 2,  3,  5,    1
        # 0948333, 2,  3,  5,    1
        # 
        self.term_frequency = csr_matrix((data, indices, indptr), dtype=np.float)

        # create a document frequency representation
        self.document_frequency = np.ndarray(shape=(1, len(self.terms)), dtype=np.float)
        for word, index in self.terms.items():
            self.document_frequency[0, index] = len(document_frequency[word])

        '''
        if normalize:
            np.apply_along_axis(lambda x : x / np.sqrt(np.sum(np.power(x, 2))), 1, self.term_frequency)
            print('Term frequency normalized')
        '''

        from math import log
        def get_inverse(x):
            return log(self.term_frequency.shape[0] / (1 +x))

        get_inverse_vectorized = np.vectorize(get_inverse)
        self.inverse_document_frequency = csr_matrix(get_inverse_vectorized(self.document_frequency))
        self.tfidf = self.term_frequency.multiply(self.inverse_document_frequency)

        word = 'antelope'
        page_id = 1369072
        page_index = page_id_to_index[page_id]
        word_index = self.terms[word]
        print('Page Index', page_index, 'Word Index', word_index)
        print('Page:', page_id, 'Term:', word) 
        print('Number of Pages containing term:', len(document_frequency[word]))
        print('Global Rarity:', self.inverse_document_frequency[0, word_index])
        print('Local frequency', self.term_frequency[page_index, word_index])
        print('Gloabl frequency:', self.document_frequency[0, word_index])
        print('TFIDF', self.tfidf[page_index, word_index])
