import numpy as np
from gensim.models import Word2Vec


def make_embedding(corpus, terms, size=100, window=5, min_count=5, workers=-1):
    '''
    Makes embeddings for terms in a corpus using SkipGram.
    Embeddings for multi-word terms are the average of embeddings for each word in the term.
    Input:
        corpus: a list of tokenized documents
        terms: a list of terms
        size: the dimensionality of the embedding
        window: the maximum distance between a target word and words around the target word
        min_count: the minimum number of occurrences of a word to be included in the vocabulary
        workers: the number of threads to use while training
    Output:
        embeddings: a dictionary of term embeddings
    '''

    model = Word2Vec(sentences=corpus, vector_size=size,
                     window=window, min_count=min_count, workers=workers, sg=1)
    embeddings = {}

    for term in terms:
        term_tokens = term.split()
        term_embedding = sum(model.wv[token]
                             for token in term_tokens) / len(term_tokens)
        embeddings[term] = term_embedding / np.linalg.norm(term_embedding)
    return embeddings
