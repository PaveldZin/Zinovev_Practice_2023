from scores import pop


class Topic:
    def __init__(self, terms=[], corpus=[], embeddings={}, scores={}):
        '''
        A Topic object is the node of the taxonomy tree.
        terms: a list of terms in the topic
        corpus: a list of documents in the topic
        embeddings: a dictionary of term embeddings
        scores: a dictionary of representativeness scores for each term
        '''
        self.terms = terms
        self.corpus = corpus
        self.embeddings = embeddings
        self.scores = scores

    def top_terms(self, n=8):
        if self.scores:
            return sorted(self.terms, key=lambda x: self.scores[x], reverse=True)[:n]
        else:
            return sorted(self.terms, key=lambda x: pop(x, self.corpus), reverse=True)[:n]
