import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from jointtsmodel.TSWE import TSWE
from jointtsmodel.utils import coherence_score_uci, coherence_score_umass
from gensim.models import KeyedVectors

data = pd.read_csv("Data/IMDb_dataset.csv")
data = data[:1000]
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
d = 300
embedding_matrix = np.zeros((X.shape[1], d))

for i, word in enumerate(vocabulary):
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]
    else:
        embedding_matrix[i] = np.zeros(d)

### model ###
K = 10
W = 1000
lambda_ = 0.1
S = 2
A = 231.15694
alpha = 50 / K
beta = 0.05
gamma = (0.05 * A) / S


TSWE_model = TSWE(embedding_dim=d, lambda_=lambda_, n_topic_components=K, n_sentiment_components=S, doc_sentiment_prior=gamma,
                  doc_sentiment_topic_prior=alpha, topic_sentiment_word_prior=beta, random_state=717, evaluate_every=2)
print("Start modeling")
TSWE_model.fit(X.toarray(), lexicon_dict, embedding_matrix, max_iter=100)

### Evaluation ###
top_words = list(TSWE_model.getTopKWords(
    vocabulary, num_words=10).values())
y_pred = TSWE_model.getSentiment(W)
print(y_pred)
y_pred = [1 if i == 0 else -1 for i in y_pred]
print(y_pred[0:20])

print("Coherence UCI {}".format(coherence_score_uci(
    X.toarray(), inv_vocabulary, top_words)))

#print("Coherence Umass{}".format(coherence_score_umass(X.toarray(), inv_vocabulary, top_words)))

print("Accuracy {}".format(accuracy_score(
    y_true=data["sentiment"][:W], y_pred=y_pred)))

#print("Hscore {}".format(Hscore(model.transform())))

print("Top words \n{}".format(top_words))
