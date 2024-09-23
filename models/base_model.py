# models/base_model.py

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass
