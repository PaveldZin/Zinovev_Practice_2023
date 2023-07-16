from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from data import Topic
from embedding import make_embedding
from scores import representativeness


def spherical_kmeans(parent_topic, k):
    '''
    Performs spherical k-means clustering on the terms in a parent topic.
    Assigns each document to the cluster with the biggest tf-idf weight of terms.
    Input:
        parent_topic: a Topic object
        k: the number of clusters
    Output:
        a list of Topic objects, one for each cluster
    '''
    documents = [' '.join(document) for document in parent_topic.corpus]

    vectorizer = TfidfVectorizer(vocabulary=parent_topic.terms)
    tfidf_matrix = vectorizer.fit_transform(documents)

    embeddings_array = [parent_topic.embeddings[term]
                        for term in parent_topic.terms]

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(embeddings_array)

    term_cluster_labels = kmeans.labels_
    document_cluster_memberships = []
    for i in range(len(documents)):
        term_weights = tfidf_matrix.getrow(i).toarray()[0]
        cluster_memberships = {}
        for j in range(len(parent_topic.terms)):
            cluster = term_cluster_labels[j]
            weight = term_weights[j]
            if cluster not in cluster_memberships:
                cluster_memberships[cluster] = 0.0
            cluster_memberships[cluster] += weight
        document_cluster_memberships.append(cluster_memberships)

    subtopics = {}
    for i, cluster_memberships in enumerate(document_cluster_memberships):
        document = documents[i]
        cluster = max(cluster_memberships, key=cluster_memberships.get)
        if cluster not in subtopics:
            subtopics[cluster] = []
        subtopics[cluster].append(document)

    child_topics = []
    for label, subtopic_documents in subtopics.items():
        subtopic_terms = [term for term, cluster_label in zip(
            parent_topic.terms, term_cluster_labels) if cluster_label == label]
        subtopic_corpus = [document.split() for document in subtopic_documents]
        subtopic_embeddings = {
            term: parent_topic.embeddings[term] for term in subtopic_terms}
        child_topics.append(
            Topic(terms=subtopic_terms, corpus=subtopic_corpus, embeddings=subtopic_embeddings, scores={}))

    return child_topics


def adaptive_clustering(parent_topic, k, delta, root=False):
    '''
    Performs adaptive clustering on the terms in a parent topic.
    Releases terms that are not representative from child topics back to parent.
    Input:
        parent_topic: a Topic object
        k: the number of clusters
        delta: the threshold for representativeness score
        root: whether the parent topic is the root topic
    Output:
        new_parent_topic: a Topic object
        child_topics: a list of Topic objects
    '''
    C_sub = parent_topic
    new_parent_topic = Topic(terms=[], corpus=parent_topic.corpus,
                             embeddings=parent_topic.embeddings, scores=parent_topic.scores)
    child_topics = []
    while True:
        child_topics = spherical_kmeans(C_sub, k)
        for i in range(len(child_topics)):
            child_topics[i].scores = {term: representativeness(
                term, child_topics, i) for term in child_topics[i].terms}
            if root:
                continue
            for term in child_topics[i].terms:
                if child_topics[i].scores[term] < delta:
                    new_parent_topic.terms.append(term)
                    child_topics[i].terms.remove(term)
                    del child_topics[i].embeddings[term]

        combined_terms = []
        combined_corpus = []
        combined_embeddings = {}
        for topic in child_topics:
            combined_terms.extend(topic.terms)
            combined_corpus.extend(topic.corpus)
            combined_embeddings.update(topic.embeddings)
        combined_topic = Topic(
            terms=combined_terms, corpus=combined_corpus, embeddings=combined_embeddings, scores={})
        if len(combined_topic.terms) == len(C_sub.terms):
            break
        else:
            C_sub = combined_topic

    # for child_topic in child_topics:
    #     child_topic.embeddings = make_embedding(child_topic.corpus, child_topic.terms)

    return new_parent_topic, child_topics
