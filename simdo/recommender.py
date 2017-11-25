from abc import ABC, abstractmethod
import logging
import random

import numpy as np
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans
)
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    TfidfTransformer
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


from decorator import log_method_calls


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def logging_config(logger, file_handler=None):
    handler = logging.FileHandler(file_handler)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaseRecommender(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


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
        self.stop_words = None

        # checked value
        self.use_cluster = use_cluster
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.use_stopword = use_stopword
        self.use_minibatch = use_minibatch

        # evaluate
        self.tfidf_model = None
        self.tfidf_matrix = None
        self.lsa_model = None
        self.lsa_matrix = None
        self.rs_model = None

        # result
        self.ind_documents = None  # return `ids` of document

        # logging
        logging_config(logger)

    def _save_model(self, model, name, path='/tmp/model/'):
        path_file_dump = path + name + ".pkl"
        logger.info("Saving %s", path_file_dump)
        joblib.dump(model, path_file_dump)

    def _stop_words(self, path_stop_words=None):
        with open(path_stop_words) as f:
            for line in f:
                self.stop_words.append(line.strip())
        return self.stop_words

    def _vectorizer(self):
        """
        Apply tf-idf (Term Frequency - Inverse Document Frequency) model
        to dataset

        Parameters
        ----------
        dataset: iterable (recommend)
            raw description, name, ..
        use_hashing: boolean, default=True
            apply hashing vectorizer model first
        use_idf: boolean, default=True
            use `idf` equation
        use_stopword: boolean, default=True
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

        self.tfidf_model = vectorizer

    def _fit_vectorizer(self, vectorizer, dataset):
        """
        Transform dataset into tf-idf model

        Parameters
        ----------
        vectorizer: `tf-idf` obj
        dataset: iterable (recommend)
        """
        # TODO
        # from itertools import tee
        # clone_dataset, _ = tee(dataset)
        clone_dataset = _get_dataset()
        self.tfidf_model = vectorizer.fit(dataset)
        self.tfidf_matrix = vectorizer.fit_transform(clone_dataset)
        del dataset, clone_dataset
        self._save_model(self.tfidf_model, "tfidf_model")
        self._save_model(self.tfidf_matrix, "tfidf_matrix")

    def _truncate_LSA(self):
        svd = TruncatedSVD(self.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        self.lsa_model = lsa.fit(self.tfidf_matrix)
        self.lsa_matrix = lsa.fit_transform(self.tfidf_matrix)
        del lsa
        self._save_model(self.lsa_model, "lsa_model")

    def _build_k_nearest_neighbors(self, poster_vect, n_recommend=20):
        """
        Unsupervised learner for implementing neighbor searches.
        Create nn model from svd matrix

        Parameters
        ----------
        poster_vect: np.array
            tf-idf matrix
        n_recommend: int, default=20
            Top n similar document evaluate by metric (default: `cosine`)
        """
        if n_recommend is None:
            n_recommend = poster_vect.shape[0]
        rs_model = NearestNeighbors(
            n_neighbors=self.n_recommend, metric=self.metric
        ).fit(poster_vect)

        return rs_model

    def _build_cluster_kmean(self, poster_vect):
        if self.lsa_matrix is None:
            logger.warning("Must fit LSA model before clustering")
            return self
        if self.minibatch:
            logger.info("Using minibatch kmean")
            km_model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=1,
                init_size=1000,
                batch_size=1000,
                verbose=self.verbose_cluster
            )
            km_model.partial_fit(poster_vect)
        else:
            logger.info("Using kmean")
            km_model = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                max_iter=100,
                n_init=1,
                verbose=self.verbose_cluster
            )
            km_model.fit(poster_vect)
        return km_model

    # TODO
    def cluster_center_space(self):
        if self.rs_model is not None and self.use_cluster:
            logger.info(
                "Top terms per cluster:",
                self.rs_model.cluster_centers_.shape
            )
            if self.n_components:
                # FIX
                original_space_centroids = self.svd.inverse_transform(
                    self.rs_model.cluster_centers_
                )
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = \
                    self.rs_model.cluster_centers_.argsort()[:, ::-1]

            terms = self.tfidf_model.get_feature_names()
            for i in range(self.n_clusters):
                print("Cluster {}:".format(i), end='')
                for ind in order_centroids[i, :25]:
                    print(' {}'.format(terms[ind]), end='')

    def _transform(self, raw_documents):
        self._vectorizer()
        # process tf_vectorizer
        self._fit_vectorizer(self.tfidf_model, raw_documents)
        # fit tf-idf vector to lsa model
        self._truncate_LSA()

    def fit(self, raw_documents=None):
        """
        Fit model: Hashing ==> Tf-idf ==> SVD ==> LSA

        Parameters
        ----------
        raw_documents: iterable (recommend)
        """
        if raw_documents is None:
            raw_documents = _get_dataset()
        # process lsa_model
        self._transform(raw_documents)
        self._save_model(self.lsa_matrix, "lsa_matrix")
        if not self.use_cluster:
            self.rs_model = self._build_k_nearest_neighbors(self.lsa_matrix)
            self._save_model(self.rs_model, "knn_dump")
        else:
            self.rs_model = self._build_cluster_kmean(self.lsa_matrix)
            self._save_model(self.rs_model, "kmean_dump")
        del self.lsa_matrix
        return self

    def transform(self, raw_document, tfidf_model=None,
                  lsa_model=None, load_model=False):
        """
        Transform single document from tf-idf vector to svd-lsa model

        Parameters
        ----------
        raw_document: str or list or iterable (recommend)
        """
        if isinstance(raw_document, str):
            raw_document = list(raw_document)

        if load_model:
            tfidf_model = joblib.load("/tmp/model/tfidf_model.pkl")
            lsa_model = joblib.load("/tmp/model/lsa_model.pkl")

        if tfidf_model is not None and lsa_model is not None:
            tfidf_vectorizer = tfidf_model.transform(raw_document)
            vectorizer = lsa_model.transform(tfidf_vectorizer)
        else:
            tfidf_vectorizer = self.tfidf_model.transform(raw_document)
            vectorizer = self.lsa_model.transform(tfidf_vectorizer)
        return vectorizer

    def evaluate(self, vectorizer, load_model=False):
        """
        Predict model - find document similarity

        Parameters
        ----------
        vectorizer: np.array
            LSA matrix
        load_model: bool, default to False
            If True, load model from pickle file
        """
        if self.rs_model is None and not load_model:
            logger.warning('Please fit model before predict')
            return self

        if not self.use_cluster:
            if load_model:
                self.rs_model = joblib.load("/tmp/model/knn_dump.pkl")

            _, recommend_index = self.rs_model.kneighbors(vectorizer)
            self.ind_documents = list(recommend_index.flatten())
        else:
            if load_model:
                self.rs_model = joblib.load("/tmp/model/kmean_dump.pkl")
            idx_cluster = self.rs_model.predict(vectorizer)
            ind_documents = np.where(self.rs_model.labels_ == idx_cluster)[0]
            self.ind_documents = random.sample(ind_documents, self.n_samples)

    # NOTE: use __enter__() and __exit__() to process context-manager (with)
    # Ex: `with RecommenderSystem as rs: ..`
    def __enter__(self):
        """
        Performs any necessary initialization, and returns a value.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Always executed to perform necessary cleanup actions.

        Parameters
        ----------
        exc_type:
            The type of the exception.
        exc_value:
            The exception instance raised.
        traceback:
            A traceback instance.
        """
        return True  # True: Suppress this exception
