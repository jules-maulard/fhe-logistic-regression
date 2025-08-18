import numpy as np

class StandardScaler:
    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean) / self.std
        return X_norm
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class MaxAbsScaler:
    def fit(self, X: np.ndarray):
        self.max_per_col = np.max(np.abs(X), axis=0)
        self.max_per_col[self.max_per_col < 1e-10] = 1.0
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_norm = X / self.max_per_col
        return X_norm
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
    def fit(self, X):
        # X = np.array(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.min_ = self.feature_range[0]
        self.max_ = self.feature_range[1]
        return self
    def transform(self, X):
        scale = (self.max_ - self.min_) / (self.data_max_ - self.data_min_)
        X_scaled = self.min_ + (X - self.data_min_) * scale
        return X_scaled
    def fit_transform(self, X):
        return self.fit(X).transform(X)