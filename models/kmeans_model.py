# models/kmeans_model.py

from models.base_model import BaseModel
from sklearn.cluster import KMeans
import joblib

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
