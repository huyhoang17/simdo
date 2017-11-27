from abc import ABC, abstractmethod


class BaseRecommender(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
