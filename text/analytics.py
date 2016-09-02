import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

class TermFreqInverseDocFreq:
    def __init__(self):
        self.terms = None
        self.term_frequency = None
        self.doc_id_to_index = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.term_count_over_iterations = None

    def create(self, corpus, text_column, normalize = True):
        data = []
        indptr = [0]
        indices = []
        self.terms = {}
        self.doc_id_to_index = {}
        self.term_count_over_iterations = []
        for doc_id, row in corpus.iterrows():
            text = str(row[text_column]).lower()
            terms = re.findall(r'\b[^\W\d_]+\b', text)
            for term in terms:
                index = self.terms.setdefault(term, len(self.terms))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            self.term_count_over_iterations.append(len(self.terms))
            self.doc_id_to_index.setdefault(doc_id, len(self.doc_id_to_index))
        
        # create a sparse matrix representation of the term frequencies
        #
        # doc-id,  a,  as, be, bovid
        # 2342342, 2,  3,  5,  4
        # 0948333, 2,  3,  5,  0
        # 
        self.term_frequency = csr_matrix((data, indices, indptr), dtype=np.float)

        # the data format above will create an entry for every instance of a term
        # so, there could be multiple values for (0, 0)
        # summing the duplicates enables us to remove the duplicate entries 
        # later we can use this matrix to derive the document frequency matrix
        self.term_frequency.sum_duplicates()

        # normalize this if the flag is set
        # each row in the resulting matrix will now have a magnitude of 1.0
        # this is an especially important step if there is a lot of disparity the length of the documents
        #
        if normalize:
            tf = self.term_frequency 
            self.term_frequency = tf.multiply(csr_matrix(1/np.sqrt(tf.multiply(tf).sum(1))))

        # create a document frequency representation
        # how many documents does each term show up in
        #
        # a,        as,    be,     bovid 
        # 23123123, 23425, 235235, 2 
        #
        self.document_frequency = self.term_frequency.astype(np.bool).sum(0)
        
        # create the inverse document frequency
        # 
        # doc-id, a,     as,     be,   bovid
        # 5234,   0.234, 0.0123, 0.01, 8.345
        self.inverse_document_frequency = csr_matrix(np.apply_along_axis(\
            lambda x : np.log(self.term_frequency.shape[0] / (1 + x)), 1, self.document_frequency))

        # create the tfidf
        self.tfidf = self.term_frequency.multiply(self.inverse_document_frequency)


    # get a single row from the tfidf
    def __getitem__(self, document_id):
        return self.tfidf[self.doc_id_to_index[document_id]]

    # get the document frequency for a single term
    def get_doc_freq(self, term):
        term_index = self.terms[term]
        return self.document_frequency[0, term_index]

    # get the local frequency for a single term 
    # if not normalized returns an integer indicating the number of times the term is in the supplied document
    # if normalized returns a floating point indicating the component 
    # value of a unit vector representing all the terms in the supplied document
    def get_local_freq(self, document_id, term):
        doc_index = self.doc_id_to_index[document_id]
        term_index = self.terms[term]
        return self.term_frequency[doc_index, term_index]

    # returns a score indicating the term's rarity within the corpus
    # the higher the value, the more rare the term
    def get_global_rarity(self, term):
        term_index = self.terms[term]
        return self.inverse_document_frequency[0, term_index]
