import logging
import random

import numpy as np
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans
)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    TfidfTransformer
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from _base import BaseRecommender
from decorator import log_method_calls


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def logging_config(logger, file_handler="spam.log"):
    handler = logging.FileHandler(file_handler)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@log_method_calls()
class RecommendSystem(BaseRecommender):
    """
    Base class to building a content-base recommender system

    Parameters
    ----------
    n_features: int or None, default=None
        Consider the top max_features ordered by `tf` across the corpus.
    n_components: int, default = 256
        Desired dimensionality of output data.
        For LSA, a value of 100 is recommended.
    n_recommend: int, default = 20
        Number of neighbors to use for kneighbors queries.
    metric: string, default "cosine"
        Metric use for distance computation
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
    use_hashing: bool, defaults to True
        Convert a collection of text documents to a matrix of token occurrences
    use_idf: bool, defaults to True
        Enable inverse-document-frequency (idf) reweighting.
    use_stopword: bool, defaults to True
    file_handler: string or None
        File handler to logging
    """

    def __init__(self, n_features=10000, n_components=256, n_recommend=20,
                 metric="cosine", use_hashing=True, use_idf=True,
                 use_stopword=True, use_minibatch=True, verbose=True,
                 n_clusters=100, use_cluster=False, n_samples=25,
                 verbose_cluster=True, file_handler=None):

        # hyperparams
        self.n_features = n_features
        self.n_components = n_components
        self.n_recommend = n_recommend
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.metric = metric
        self.verbose = verbose
        self.verbose_cluster = verbose_cluster
        self.stop_words = []

        # checked value
        self.use_cluster = use_cluster
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.use_stopword = use_stopword
        self.use_minibatch = use_minibatch

        # evaluate
        self.tfidf_model = None
        self.lsa_model = None
        self.rs_model = None

        # result
        self.ind_documents = None  # return `ids` of document

        # logging
        logging_config(logger)

    def _stop_words(self, path_stop_words=None):
        with open(path_stop_words) as f:
            for line in f:
                self.stop_words.append(line.strip())

    def _vectorizer(self, dataset):
        """
        Apply tf-idf (Term Frequency - Inverse Document Frequency) model
        to dataset

        Parameters
        ----------
        dataset: iterable (recommend)
            raw description, name, ..
        """
        if self.use_stopword:
            self._stop_words()
        if self.use_hashing:
            if self.use_idf:
                hasher = HashingVectorizer(
                    n_features=self.n_features,
                    stop_words=self.stop_words,
                    alternate_sign=False,
                    norm=None, binary=False
                )
                vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                vectorizer = HashingVectorizer(
                    n_features=self.n_features,
                    stop_words=self.stop_words,
                    alternate_sign=False, norm='l2',
                    binary=False
                )
        else:
            vectorizer = TfidfVectorizer(
                max_df=0.5, max_features=self.n_features,
                min_df=2, stop_words=self.stop_words,
                use_idf=self.use_idf
            )

        tfidf_matrix = vectorizer.fit_transform(dataset)
        self.tfidf_model = vectorizer

        svd = TruncatedSVD(self.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        self.lsa_model = lsa.fit(tfidf_matrix)
        lsa_matrix = lsa.fit_transform(tfidf_matrix)

        return lsa_matrix

    def _build_k_nearest_neighbors(self, poster_vect):
        """
        Unsupervised learner for implementing neighbor searches.
        Create nn model from svd matrix

        Parameters
        ----------
        poster_vect: np.array
            tf-idf matrix
        """
        if poster_vect is None:
            raise NotImplementedError(
                "Must fit LSA model before implement neighbor searches"
            )
        rs_model = NearestNeighbors(
            n_neighbors=self.n_recommend, metric=self.metric
        ).fit(poster_vect)

        return rs_model

    def _build_cluster_kmean(self, poster_vect):
        """
        Unsupervised learner for implementing neighbor searches.
        Create nn model from svd matrix

        Parameters
        ----------
        poster_vect: np.array
            tf-idf matrix
        """
        if poster_vect is None:
            raise NotImplementedError("Must fit LSA model before clustering")
        if self.use_minibatch:
            logger.info("==> Using Minibatch KMean Cluster")
            rs_model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=1,
                init_size=1000,
                batch_size=1000,
                verbose=self.verbose_cluster
            )
            rs_model.partial_fit(poster_vect)
        else:
            logger.info("==> Using KMean Cluster")
            rs_model = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                max_iter=100,
                n_init=1,
                verbose=self.verbose_cluster
            )
            rs_model.fit(poster_vect)
        return rs_model

    def fit(self, raw_documents=None):
        """
        Fit model: Hashing ==> Tf-idf ==> SVD ==> LSA

        Parameters
        ----------
        raw_documents: iterable (recommend)
        """
        lsa_matrix = self._vectorizer(raw_documents)
        if not self.use_cluster:
            self.rs_model = self._build_k_nearest_neighbors(lsa_matrix)
        else:
            self.rs_model = self._build_cluster_kmean(lsa_matrix)
        return self

    def transform(self, raw_document):
        """
        Transform single document from tf-idf vector to SVD-LSA model

        Parameters
        ----------
        raw_document: str or list or iterable (recommend)
        """
        if isinstance(raw_document, str):
            raw_document = list(raw_document)

        tfidf_vectorizer = self.tfidf_model.transform(raw_document)
        vectorizer = self.lsa_model.transform(tfidf_vectorizer)
        return vectorizer

    def evaluate(self, vectorizer):
        """
        Predict model - find document similarity

        Parameters
        ----------
        vectorizer: np.array
            LSA matrix
        """
        if self.use_cluster:
            idx_cluster = self.rs_model.predict(vectorizer)
            ind_documents = list(np.where(
                self.rs_model.labels_ == idx_cluster
            )[0])
            self.ind_documents = random.sample(
                ind_documents, self.n_samples
            )
        else:
            _, recommend_index = self.rs_model.kneighbors(vectorizer)
            self.ind_documents = list(recommend_index.flatten())

        return self.ind_documents
