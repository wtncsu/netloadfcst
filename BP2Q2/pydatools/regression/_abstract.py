#
# Abstract classes for regression models
#

from abc import (
    ABC,
    abstractmethod,
)



class Regressor(ABC):
    @abstractmethod
    def fit(self, features, target):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, features):
        raise NotImplementedError()
