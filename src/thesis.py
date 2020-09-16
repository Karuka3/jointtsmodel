import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from jointtsmodel.TSWE import TSWE
from jointtsmodel.utils import coherence_score_umass

data = pd.read_csv("Data/IMDb_dataset.csv")
vectorizer = CountVectorizer(max_df=0.5, min_df=15, stop_words='english')
X = vectorizer.fit_transform(data["review"])
vocabulary = vectorizer.get_feature_names()
inv_vocabulary = dict(zip(vocabulary, np.arange(len(vocabulary))))

lexicon_data = pd.read_csv('lexicon/prior_sentiment.csv')
lexicon_data = lexicon_data.dropna()
lexicon_dict = dict(zip(lexicon_data['Word'], lexicon_data['Sentiment']))

### Load word embeddings for TSWE model ###
embeddings_index = {}
f = open('Data/glove.6B.100d.txt', 'r', encoding='utf8')

for i, line in enumerate(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((X.shape[1], 100))

for i, word in enumerate(vocabulary):
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]
    else:
        embedding_matrix[i] = np.zeros(100)

### TSWE model ###
K = 5
S = 2
A = 231.15694
alpha = 50 / K
beta = 0.05
gamma = (0.05 * A) / S

model = TSWE(embedding_dim=100, n_topic_components=K, n_sentiment_components=S, doc_sentiment_prior=alpha,
             doc_sentiment_topic_prior=gamma, topic_sentiment_word_prior=beta, random_state=123, evaluate_every=2)
print("Start modeling")
model.fit(X.toarray(), lexicon_dict, embedding_matrix, max_iter=1)

### Evaluation ###
top_words = list(model.getTopKWords(vocabulary).values())
print("Coherence {}".format(coherence_score_uci(
    X.toarray(), inv_vocabulary, top_words)))
print("Hscore {}".format(Hscore(model.transform())))
print("Top words \n{}".format(top_words))
