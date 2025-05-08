import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train-x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        return np.argmax(np.bincount(k_labels))

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
