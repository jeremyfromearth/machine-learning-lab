import re
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

class TermFreqInverseDocFreq:
    def __init__(self):
        self.terms = None
        self.term_frequency = None
        self.term_id_to_term = None
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
        self.term_id_to_term = {}
        self.term_count_over_iterations = []
        for doc_id, row in corpus.iterrows():
            text = str(row[text_column]).lower()
            terms = re.findall(r'\b[^\W\d_]+\b', text)
            for term in terms:
                index = self.terms.setdefault(term, len(self.terms))
                self.term_id_to_term.setdefault(index, term)
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            self.term_count_over_iterations.append(len(self.terms))
            self.doc_id_to_index.setdefault(doc_id, len(self.doc_id_to_index))
        
        # Create a sparse matrix representation of the term frequencies
        #
        # doc-id,  a,  as, be, bovid
        # 2342342, 2,  3,  5,  4
        # 0948333, 2,  3,  5,  0
        # 
        self.term_frequency = csr_matrix((data, indices, indptr), dtype=np.float)

        # The data format above will create an entry for every instance of a term
        # so, there could be multiple values for the term at (0, 0).
        # Summing the duplicates removes the duplicate entries.
        # This enables deriving the document frequency matrix from the term_frequency matrix
        self.term_frequency.sum_duplicates()

        # Normalize this if the flag is set.
        # Each row in the resulting matrix will now have a magnitude of 1.0.
        # This is an especially important step if there is a lot of disparity the length of the documents.
        if normalize:
            tf = self.term_frequency 
            self.term_frequency = tf.multiply(csr_matrix(1/np.sqrt(tf.multiply(tf).sum(1))))

        # Create a document frequency representation
        #
        # a,        as,    be,     bovid 
        # 23123123, 23425, 235235, 2 
        #
        self.document_frequency = self.term_frequency.astype(np.bool).sum(0)
        
        # Create the inverse document frequency
        # 
        # doc-id, a,     as,     be,   bovid
        # 5234,   0.234, 0.0123, 0.01, 8.345
        self.inverse_document_frequency = csr_matrix(np.apply_along_axis(
            lambda x : np.log(self.term_frequency.shape[0] / (1 + x)), 1, self.document_frequency))

        # Create the tfidf
        self.tfidf = self.term_frequency.multiply(self.inverse_document_frequency)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'terms': self.terms,
                'term_frequency': self.term_frequency,
                'doc_id_to_index': self.doc_id_to_index,
                'term_id_to_term': self.term_id_to_term,
                'inverse_document_frequency': self.inverse_document_frequency,
                'document_frequency': self.document_frequency
            }, f)
                
    def load(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            self.tfidf = obj['tfidf']
            self.terms = obj['terms']
            self.term_frequency = obj['term_frequency']
            self.doc_id_to_index = obj['doc_id_to_index']
            self.term_id_to_term = obj['term_id_to_term']
            self.inverse_document_frequency = obj['inverse_document_frequency']
            self.document_frequency = obj['document_frequency']

    # Get a single row from the tfidf
    def __getitem__(self, document_id):
        if document_id not in self.doc_id_to_index:
            return None
        return self.tfidf[self.doc_id_to_index[document_id]]

    # Get the document frequency for a single term
    def get_doc_freq(self, term):
        if term not in self.terms:
            return 0
        term_index = self.terms[term]
        return self.document_frequency[0, term_index]

    # Get the local frequency for a single term 
    # If not normalized returns an integer indicating the number of times the term is in the supplied document
    # If normalized returns a floating point indicating the component 
    # Value of a unit vector representing all the terms in the supplied document
    def get_local_freq(self, document_id, term):
        if document_id not in self.doc_id_to_index or term not in self.terms:
            return 0.0
        doc_index = self.doc_id_to_index[document_id]
        term_index = self.terms[term]
        return self.term_frequency[doc_index, term_index]

    # Returns a score indicating the term's rarity within the corpus
    # The higher the value, the more rare the term
    def get_global_rarity(self, term):
        if term not in self.terms: 
            return 0.0
        term_index = self.terms[term]
        return self.inverse_document_frequency[0, term_index]

    def get_sorted_terms_for_document(self, document_id):
        terms = {}
        if document_id not in self.doc_id_to_index:
            return pd.Series()
        doc_index = self.doc_id_to_index[document_id]
        tf = self.tfidf[doc_index].toarray()
        for index in range(len(tf[0])):
            value = tf[0, index]
            terms[self.term_id_to_term[index]] = value if value >= 0.0 else 0.0
        result = pd.Series(terms)
        result.sort_values(inplace=True, ascending=False)
        return result
