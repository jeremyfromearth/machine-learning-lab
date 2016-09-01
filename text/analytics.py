import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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
        vocab = {} 
        data = []
        indptr = [0]
        indices = []
        page_id_to_index = {}
        for page_id, row in corpus.iterrows():
            s = str(row[text_column]).lower()
            word_list = re.findall(r'\b[^\W\d_]+\b', s)
            page_id_to_index.setdefault(page_id, len(page_id_to_index))
            for word in word_list:
                index = vocab.setdefault(word, len(vocab))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            print(page_id_to_index[page_id], '-', page_id)
        sparse = csr_matrix((data, indices, indptr), dtype=np.float)

        # Test should print 2.0
        page_index = page_id_to_index[1709509]
        word_index = vocab['tagua']
        print(sparse[page_index, word_index])

        '''
        text = {}
        unique_words = set()
        for page_id, row in corpus.iterrows():
            s = str(row[text_column]).lower()
            word_list = re.findall(r'\b[^\W\d_]+\b', s) 
            text[page_id] = word_list

            for word in word_list:
                unique_words.add(word)

            self.word_count_over_iterations.append(len(unique_words))

        self.unique_words = sorted(list(unique_words))
        self.term_frequency = np.zeros(shape=(len(text), len(self.unique_words)), dtype=np.float)
        reverse_word_lookup = dict({(self.unique_words[i], i) for i in range(len(self.unique_words))}) 

        n = 0
        self.reverse_page_id_lookup = {}
        for page_id, words in text.items():
            #terms = np.zeros(len(self.unique_words))
            self.reverse_page_id_lookup[page_id] = n
            for word in words:
                self.term_frequency[n][reverse_word_lookup[word]] += 1
            print('Page', page_id, '-', n, 'processed')
            n += 1
        '''
        

        '''
        sparse = csr_matrix(self.term_frequency, dtype=np.float)
        print(sys.getsizeof(self.term_frequency))
        print(sys.getsizeof(sparse))

        page_id = 3451749
        page_index = self.reverse_page_id_lookup[page_id]
        word = 'foothill'
        word_index = reverse_word_lookup[word]
        print(sparse[page_index, word_index])

        if normalize:
            np.apply_along_axis(lambda x : x / np.sqrt(np.sum(np.power(x, 2))), 1, self.term_frequency)
            print('Term frequency normalized')
        
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
