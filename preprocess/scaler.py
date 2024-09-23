# preprocess/scaler.py

from sklearn.preprocessing import StandardScaler
import pandas as pd

class Scaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def save_scaler(self, path):
        import joblib
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        import joblib
        self.scaler = joblib.load(path)
