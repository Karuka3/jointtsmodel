"""
(c) Ayan Sengupta - 2020
License: MIT License

Implementation of WS-TSWE (Weakly Supervised Topic-Sentiment model with Word Embeddings)

Reference
    [1] http://ceur-ws.org/Vol-1646/paper6.pdf

"""

# Author: Ayan Sengupta

#import warnings
#from collections import defaultdict
#import inspect
from __future__ import absolute_import
from sklearn.utils.validation import check_is_fitted, check_non_negative, check_random_state, check_array
import numpy as np
import scipy
from tqdm import tqdm
from scipy.special import gammaln, psi
from scipy.optimize import minimize
from .base import BaseEstimator
from .utils import sampleFromDirichlet, sampleFromCategorical, log_multi_beta, word_indices
from .utils import coherence_score_uci, coherence_score_umass, symmetric_kl_score, Hscore


def L(v_k, embedding_matrix, N_k, mu=.01):
    factor1 = np.log(
        np.sum(np.exp(np.dot(v_k, embedding_matrix.T)))) * np.sum(N_k)
    factor2 = np.dot(np.dot(v_k, embedding_matrix.T), N_k)
    return (mu*np.linalg.norm(v_k)) - factor2 + factor1


class TSWE(BaseEstimator):
    """TSWE model

    Parameters
    ----------
    embedding_dim : int, optional (default=300)
        Dimension of Word Embeddings and Topic Embeddings
    lambda_ : float, optional (default=.1)
        Lambda parameter controls the effect of word embeddings in the model. Higher lambda_ value denotes higher effect of embeddings.
    n_topic_components : int, optional (default=10)
        Number of topics.
    n_sentiment_components : int, optional (default=5)
        Number of sentiments.
    doc_sentiment_prior : float, optional (default=None)
        Prior of document sentiment distribution `theta`. If the value is None,
        defaults to `1 / n_sentiment_components`.
    doc_sentiment_topic_prior : float, optional (default=None)
        Prior of document topic-sentiment distribution `pi`. If the value is None,
        defaults to `1 / n_topic_components`.
    topic_sentiment_word_prior : float, optional (default=None)
        Prior of topic-sentiment word distribution `beta`. If the value is None, defaults
        to `1 / (n_topic_components * n_sentiment_components)`.
    max_iter : integer, optional (default=10)
        The maximum number of iterations for Gibbs sampling.
    prior_update_step: integer, optional (default=5)
        How often to update priors using Minka's fixed point iteration
    evaluate_every : int, optional (default=0)
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.
    verbose : int, optional (default=0)
        Verbosity level.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    components_ : array, [self.vocabSize, n_topic_components, n_sentiment_components]
        topic-sentiment word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j, k]`` can be viewed as pseudocount that represents the
        number of times word `i` was assigned to topic `j` and sentiment `k`.
        It can also be viewed as distribution over the words for each topic-sentiment pair
        after normalization:
        ``model.components_ / model.components_.sum(axis=0)[np.newaxis,:,:]``.
    doc_sentiment_prior_ : float
        Prior of document sentiment distribution `pi`. If the value is None,
        it is `1 / n_sentiment_components`.
    doc_sentiment_topic_prior_ : float
        Prior of document-sentiment-topics distribution `theta`. If the value is None,
        it is `1 / n_topic_components`.
    topic_sentiment_word_prior_ : float
        Prior of topic-sentiment-word distribution `beta`. If the value is None, it is
        `1 / (n_topic_components * n_sentiment_components)`.

    Reference
    ---------
        [1] http://ceur-ws.org/Vol-1646/paper6.pdf

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    def __init__(self, embedding_dim=300, lambda_=0.1, n_topic_components=10, n_sentiment_components=5, doc_topic_prior=None, doc_sentiment_prior=None,
                 doc_topic_sentiment_prior=None, doc_sentiment_topic_prior=None,
                 topic_sentiment_word_prior=None, max_iter=10,
                 prior_update_step=2, evaluate_every=1, verbose=1, random_state=None):

        super().__init__(n_topic_components=n_topic_components, n_sentiment_components=n_sentiment_components, doc_topic_prior=doc_topic_prior, doc_sentiment_prior=doc_sentiment_prior,
                         doc_topic_sentiment_prior=doc_topic_sentiment_prior, doc_sentiment_topic_prior=doc_sentiment_topic_prior,
                         topic_sentiment_word_prior=topic_sentiment_word_prior, max_iter=max_iter,
                         prior_update_step=prior_update_step, evaluate_every=evaluate_every, verbose=verbose, random_state=random_state)

        self.embedding_dim = embedding_dim
        self.lambda_ = lambda_

    def _initialize_(self, X, lexicon_dict, word_embedding_matrix):
        """Initialize fit variables
        Parameters
        ----------
        X : array-like, shape=(n_docs, self.vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        Returns
        -------
        self
        """

        self.wordOccurenceMatrix = X
        self._check_params()
        self._init_latent_vars()
        self.word_embeddings = word_embedding_matrix

        n_docs, self.vocabSize = self.wordOccurenceMatrix.shape

        # Pseudocounts
        self.n_ds = np.zeros((n_docs, self.n_sentiment_components))
        self.n_dst = np.zeros(
            (n_docs, self.n_sentiment_components, self.n_topic_components))
        self.n_d = np.zeros((n_docs))
        self.n_vts = np.zeros(
            (self.vocabSize, self.n_topic_components, self.n_sentiment_components))
        self.n_ts = np.zeros(
            (self.n_topic_components, self.n_sentiment_components))
        self.n_vt = np.zeros((self.vocabSize, self.n_topic_components))

        self.topic_embeddings = np.zeros(
            (self.n_topic_components, self.embedding_dim))

        self.topics = {}
        self.sentiments = {}
        self.alphaVec = self.doc_sentiment_topic_prior_
        self.gammaVec = self.doc_sentiment_prior_.copy()
        self.beta = self.topic_sentiment_word_prior_

        for d in range(n_docs):
            sentimentDistribution = sampleFromDirichlet(self.gammaVec)
            topicDistribution = np.zeros(
                (self.n_sentiment_components, self.n_topic_components))

            for s in range(self.n_sentiment_components):
                topicDistribution[s, :] = sampleFromDirichlet(self.alphaVec)

            for i, w in enumerate(word_indices(self.wordOccurenceMatrix[d, :])):
                s = sampleFromCategorical(sentimentDistribution)
                t = sampleFromCategorical(topicDistribution[s, :])

                prior_sentiment = lexicon_dict.get(w, 1)

                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_ds[d, s] += 1
                self.n_dst[d, s, t] += 1
                self.n_d[d] += 1
                self.n_ts[t, s] += 1
                self.n_vts[w, t, s*prior_sentiment] += 1
                self.n_vt[w, t] += 1

    def conditionalDistribution(self, d, v):
        """
        Calculates the joint topic-sentiment probability for word v in document d
        Parameters
        -----------
        d: index
            Document index
        v: index
            Word index
        Returns
        ------------
        x: matrix
            Matrix (n_topic_components x n_sentiment_components) of joint probabilities
        """
        probabilities_ts = np.ones(
            (self.n_topic_components, self.n_sentiment_components))

        firstFactor = (self.n_ds[d] + self.gammaVec) / \
            (self.n_d[d] + np.sum(self.gammaVec))
        secondFactor = np.zeros(
            (self.n_topic_components, self.n_sentiment_components))
        for s in range(self.n_sentiment_components):
            secondFactor[:, s] = (
                (self.n_dst[d, s, :] + self.alphaVec) / (self.n_ds[d, s] + np.sum(self.alphaVec)))

        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)

        #forthFactor = np.zeros((self.n_topic_components, self.n_sentiment_components))
        # for k in range(self.n_topic_components):
        #    forthFactor[k,:] = np.exp(np.dot(self.topic_embeddings[k,:],self.word_embeddings[v,:]))/np.sum(np.exp(np.dot(self.topic_embeddings[k,:],self.word_embeddings.T)))

        forthFactor = np.exp(np.dot(self.topic_embeddings, self.word_embeddings[v, :])) / np.sum(
            np.exp(np.dot(self.topic_embeddings, self.word_embeddings.T)), -1)
        probabilities_ts *= firstFactor[np.newaxis, :]
        probabilities_ts *= secondFactor
        probabilities_ts *= ((1 - self.lambda_)*thirdFactor +
                             (self.lambda_*forthFactor[:, np.newaxis]))
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def fit(self, X, lexicon_dict, word_embedding_matrix, rerun=False, max_iter=None):
        """Learn model for the data X with Gibbs sampling.
        Parameters
        ----------
        X : array-like, shape=(n_docs, self.vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        rerun: bool (default=False)
            If True then we do not re initialize the model
        max_iter : int, optional (default=None)
        Returns
        -------
        self
        """
        if rerun == False:
            self._initialize_(X, lexicon_dict, word_embedding_matrix)

        self.wordOccurenceMatrix = self._check_non_neg_array(
            self.wordOccurenceMatrix, "TSWE.fit")
        if max_iter is None:
            max_iter = self.max_iter

        self.all_loglikelihood = []
        self.all_perplexity = []
        n_docs, self.vocabSize = self.wordOccurenceMatrix.shape
        for iteration in tqdm(range(max_iter)):
            for d in range(n_docs):
                for i, v in enumerate(word_indices(self.wordOccurenceMatrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_ds[d, s] -= 1
                    self.n_d[d] -= 1
                    self.n_dst[d, s, t] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1
                    self.n_vt[v, t] -= 1

                    probabilities_ts = self.conditionalDistribution(d, v)
                    if v in lexicon_dict:
                        s = lexicon_dict[v]
                        t = sampleFromCategorical(probabilities_ts[:, s])
                    else:
                        a = probabilities_ts.flatten()
                        ind = np.argmax(a)
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_d[d] += 1
                    self.n_dst[d, s, t] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
                    self.n_ds[d, s] += 1
                    self.n_vt[v, t] += 1

            '''
            if self.prior_update_step > 0 and (iteration+1)%self.prior_update_step == 0:
                numerator = 0
                denominator = 0
                for d in range(n_docs):
                    numerator += psi(self.n_d[d] + self.alphaVec) - psi(self.alphaVec)
                    denominator += psi(np.sum(self.n_ds[d] + self.alphaVec)) - psi(np.sum(self.alphaVec))

                self.alphaVec *= numerator / denominator
            '''
            if self.prior_update_step > 0 and (iteration + 1) % self.prior_update_step == 0:
                #print("Updating topic embeddings")
                for k in range(self.n_topic_components):
                    res = minimize(L, self.topic_embeddings[k, :], method='L-BFGS-B', args=(
                        self.word_embeddings, self.n_vt[:, k]))
                    self.topic_embeddings[k] = res.x

            #loglikelihood_ = self.loglikelihood()
            #perplexity_ = self.perplexity()

            # if self.evaluate_every > 0 and (iteration+1)%self.evaluate_every == 0:
            #    if self.verbose > 0:
            #        print ("Perplexity after iteration {} (out of {} iterations) is {:.2f}".format(iteration + 1, max_iter, perplexity_))

        self.doc_sentiment_prior_ = self.gammaVec
        normalized_n_vts = self.n_vts.copy() + self.beta
        normalized_n_vts /= normalized_n_vts.sum(0)[np.newaxis, :, :]
        self.components_ = normalized_n_vts

        return self

    def _unnormalized_transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_sentiment_distr : shape=(n_docs, n_sentiment_components)
            Document sentiment distribution for X.
        """
        return self.n_ds + self.doc_sentiment_prior_

    def transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_sentiment_distr : shape=(n_docs, n_sentiment_components)
            Document sentiment distribution for X.
        """
        normalize_n_ds = self._unnormalized_transform().copy()
        normalize_n_ds /= normalize_n_ds.sum(1)[:, np.newaxis]
        return normalize_n_ds

    def fit_transform(self, X, lexicon_dict, rerun=False, max_iter=None):
        """Fit and transform data according to fitted model.
        Parameters
        ----------
        X : array-like, shape=(n_docs, self.vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        rerun: bool (default=False)
            If True then we do not re initialize the model
        max_iter : int, optional (default=None)
        Returns
        -------
        doc_sentiment_distr : shape=(n_samples, n_sentiment_components)
            Document sentiment distribution for X.
        """
        return self.fit(X, lexicon_dict, rerun=rerun, max_iter=max_iter).transform()

    def pi(self):
        return self.transform()

    def theta(self):
        """Document-sentiment-topic distribution according to fitted model.
        Returns
        -------
        doc_sentiment_topic_dstr : shape=(n_docs, n_sentiment_components, n_topic_components)
            Document-sentiment-topic distribution for X.
        """
        normalized_n_dst = self.n_dst.copy() + self.gammaVec
        normalized_n_dst /= normalized_n_dst.sum(2)[:, :, np.newaxis]
        return normalized_n_dst

    def loglikelihood(self):
        """Calculate log-likelihood of generating the whole corpus
        Returns
        -----------
        Log-likelihood score: float
        """
        score = 0
        raise NotImplementedError("To be implemented")

    def perplexity(self):
        """Calculate approximate perplexity for the whole corpus.
        Perplexity is defined as exp(-1. * log-likelihood per word)

        Returns
        ------------
        score : float
        """
        score = np.exp((-1.) * self.loglikelihood() /
                       np.sum(self.wordOccurenceMatrix))
        return score

    def getSentiment(self):
        normalized_n_ds = self.transform()
        y_pred = []
        for d in range(self.wordOccurenceMatrix.shape[0]):
            max_index = np.argmax(normalized_n_ds[d])
            y_pred.append(max_index)
        return y_pred
