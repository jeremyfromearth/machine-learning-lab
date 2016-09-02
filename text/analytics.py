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
        print('TermFreqInverseDocFreq.create()')
        data = []
        indptr = [0]
        indices = []
        self.terms = {}
        page_id_to_index = {}
        for page_id, row in corpus.iterrows():
            text = str(row[text_column]).lower()
            terms = re.findall(r'\b[^\W\d_]+\b', text)
            page_id_to_index.setdefault(page_id, len(page_id_to_index))
            for word in terms:
                index = self.terms.setdefault(word, len(self.terms))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            self.word_count_over_iterations.append(len(self.terms))
        
        # create a sparse matrix representation of the term frequencies
        #
        # page-id, a, as, be, both
        # 2342342, 2,  3,  5,    1
        # 0948333, 2,  3,  5,    1
        # 
        self.term_frequency = csr_matrix((data, indices, indptr), dtype=np.float)
        self.term_frequency.sum_duplicates()
        print('Term frequency created')

        if normalize:
            tf = self.term_frequency 
            self.term_frequency = tf.multiply(csr_matrix(1/np.sqrt(tf.multiply(tf).sum(1))))
            print('Term frequency normalized')

        # create a document frequency representation
        self.document_frequency = self.term_frequency.astype(np.bool).sum(0)

        self.inverse_document_frequency = csr_matrix(np.apply_along_axis(\
            lambda x : np.log(self.term_frequency.shape[0] / (1 + x)), 1, self.document_frequency))
        self.tfidf = self.term_frequency.multiply(self.inverse_document_frequency)
        print('TFIDF created')

        word = 'antelope'
        page_id = 1369072
        page_index = page_id_to_index[page_id]
        word_index = self.terms[word]
        print('Page Index', page_index, 'Word Index', word_index)
        print('Page:', page_id, 'Term:', word) 
        print('Global Rarity:', self.inverse_document_frequency[0, word_index])
        print('Local frequency', self.term_frequency[page_index, word_index])
        print('Gloabl frequency:', self.document_frequency[0, word_index])
        print('TFIDF', self.tfidf[page_index, word_index])
