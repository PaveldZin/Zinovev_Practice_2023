import json
from collections import Counter

import spacy

nlp = spacy.load('en_core_web_sm')


def load_json(filename, categories=None):
    '''
    Load json file and return a list of papers dictionaries.
    Input:
        filename: the name of the json file
        categories: a list of Arxiv categories to filter papers by
    Output:
        papers: a list of papers dictionaries
    '''
    papers = []
    with open(filename, 'r') as f:
        for line in f:
            papers.append(json.loads(line))
    if categories:
        papers = [paper for paper in papers if all(
            category in paper['categories'] for category in categories)]
    return papers


def extract_terms_and_corpus(papers, threshold=5):
    '''
    Extract noun phrases from abstracts and return a list of terms and a tokenized corpus.
    Input:
        papers: a list of papers dictionaries
        threshold: the minimum frequency of a noun phrase to be included in the list of terms
    Output:
        terms: a list of terms
        corpus: a tokenized corpus
    '''
    noun_phrases = []
    corpus = []
    for paper in papers:
        paper['abstract'] = paper['abstract'].replace("\n", " ")
        paper['abstract'] = paper['abstract'].lower()
        doc = nlp(paper['abstract'])
        tokens = [token.text for token in doc if not (
            token.is_stop or token.is_punct or token.is_space)]
        corpus.append(tokens)
        for chunk in doc.noun_chunks:
            phrase = ' '.join(token.text for token in chunk if not (
                token.is_stop or token.is_punct or token.is_space))
            if phrase:
                noun_phrases.append(phrase)

    freq = Counter(noun_phrases)

    terms = list(
        set([phrase for phrase in freq if freq[phrase] >= threshold]))

    return terms, corpus


def extract_corpus(papers):
    '''
    Extract corpus from abstracts and return a tokenized corpus.
    Input:
        papers: a list of papers dictionaries
    Output:
        corpus: a tokenized corpus
    '''
    corpus = []
    for paper in papers:
        paper['abstract'] = paper['abstract'].replace("\n", " ")
        paper['abstract'] = paper['abstract'].lower()
        doc = nlp(paper['abstract'])
        tokens = [token.text for token in doc if not (
            token.is_stop or token.is_punct or token.is_space)]
        corpus.append(tokens)
    return corpus
