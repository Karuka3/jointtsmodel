import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from jointtsmodel.TSWE import TSWE
from jointtsmodel.utils import coherence_score_umass
from gensim.models import KeyedVectors


def TSWE_modeling(K, X, d, lambda_, S, alpha, beta, gamma, lexicon_dict, embedding_matrix, data, vocabulary, inv_vocabulary, save=False):
    TSWE_model = TSWE(embedding_dim=d, lambda_=lambda_, n_topic_components=K, n_sentiment_components=S, doc_sentiment_prior=gamma,
                      doc_sentiment_topic_prior=alpha, topic_sentiment_word_prior=beta, random_state=717, evaluate_every=2)
    TSWE_model.fit(X.toarray(), lexicon_dict, embedding_matrix, max_iter=100)

    print("Topic K: {}".format(K))
    accuracy = eval_accuracy(TSWE_model, y_true=data["sentiment"][:X.shape[0]])
    coherence, top_words = eval_coherence(
        TSWE_model, X, vocabulary, inv_vocabulary)
    if save:
        saving = {"Accuracy": accuracy, "Coherence": coherence,
                  "TopWords": top_words, "model": TSWE_model}
        name = "Topic{}_TSWEmodel.pickle".format(K)
        with open(name, mode='wb') as f:
            pickle.dump(saving, f)


def wrapper(args):
    return TSWE_modeling(*args)


def multi_process(args_list):
    print("CPU_COUNT {}".format(cpu_count()))
    with Pool(24) as p:
        p.map(wrapper, args_list)


def data_preprocessing(data):
    text_data = data
    https = re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+")
    http = re.compile(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-…]+")
    alphabet = re.compile(r"[^a-zA-Z.?' ]")
    text_data = [https.sub("", text) for text in text_data]
    text_data = [http.sub("", text) for text in text_data]
    text_data = [BeautifulSoup(text, 'html.parser') for text in text_data]
    text_data = [text.get_text() for text in text_data]
    text_data = [alphabet.sub("", text) for text in text_data]
    pre_data = [text.lower() for text in text_data]
    return pre_data


def eval_accuracy(model, y_true):
    y_pred = model.getSentiment()
    y_pred = [1 if i == 0 else -1 for i in y_pred]
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    if accuracy < 0.5:
        accuracy = (1 - accuracy)
    print("Accuracy: {}".format(accuracy))
    return accuracy


def eval_coherence(model, X, vocabulary, inv_vocabulary):
    top_words = list(model.getTopKWords(vocabulary, num_words=15).values())
    coherence = coherence_score_umass(
        X.toarray(), inv_vocabulary, top_words, normalized=True)
    print("Coherence UMass: {}".format(coherence))
    print("Top words \n{}".format(pd.DataFrame(top_words).transpose()))
    return coherence, top_words


if __name__ == "__main__":
    df = pd.read_csv("Data/IMDb_dataset.csv")
    W = 2000
    data = df[:W].copy()
    data["review"] = data_preprocessing(data["review"])
    del df

    vectorizer = CountVectorizer(max_df=15, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(data["review"])
    vocabulary = vectorizer.get_feature_names()
    inv_vocabulary = dict(zip(vocabulary, np.arange(len(vocabulary))))

    lexicon_data = pd.read_csv('lexicon/prior_sentiment.csv')
    lexicon_data = lexicon_data.dropna()
    lexicon_dict = dict(zip(lexicon_data['Word'], lexicon_data['Sentiment']))

    ### Load word embeddings for TSWE model ###
    embeddings_index = KeyedVectors.load_word2vec_format(
        "Data/GoogleNews-vectors-negative300.bin", binary=True)
    dimension = 300
    embedding_matrix = np.zeros((X.shape[1], dimension))

    for i, word in enumerate(vocabulary):
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
        else:
            embedding_matrix[i] = np.zeros(dimension)

    ### model parameter ###
    Ks = [1, 5, 10, 20]
    lambda_ = 0.1
    S = 2
    A = np.sum(X.toarray()) / W
    # alpha = 50 / K
    beta = 0.05
    gamma = (0.05 * A) / S

    args_list = [(K, X, dimension, lambda_, S, 50/K, beta, gamma, lexicon_dict,
                  embedding_matrix, data, vocabulary, inv_vocabulary, True) for K in Ks]
    multi_process(args_list)
