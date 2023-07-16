import math

from rank_bm25 import BM25Okapi


def pop(term, corpus):
    '''
    Compute the popularity score of a term in a corpus.
    '''
    term_count = 0
    total_tokens = 0
    for document in corpus:
        term_count += ' '.join(document).count(term)
        total_tokens += len(document)
    return math.log(term_count + 1) / math.log(total_tokens)


def con(term, child_topics, main_topic_index):
    '''
    Compute concentration score of a term in a topic.
    '''
    joined_documents = []
    for i in range(len(child_topics)):
        joined_document = [
            element for sublist in child_topics[i].corpus for element in sublist]
        joined_documents.append(joined_document)
    bm25 = BM25Okapi(joined_documents)
    tokenized_query = term.split()
    doc_scores = bm25.get_scores(tokenized_query)
    return math.exp(doc_scores[main_topic_index]) / (1 + sum(math.exp(doc_score) for doc_score in doc_scores))


def representativeness(term, child_topics, main_topic_index):
    '''
    Compute the representativeness score of a term in a topic.
    '''
    return math.sqrt(pop(term, child_topics[main_topic_index].corpus) * con(term, child_topics, main_topic_index))
