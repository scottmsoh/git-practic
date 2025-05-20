import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = self._compute_distances(X)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        y_pred = np.array([np.argmax(np.bincount(row)) for row in k_labels])

        return y_pred

    def _compute_distances(self, X):
        X_square = np.sum(X**2, axis=1, keepdims=True)
        train_square = np.sum(self.X_train**2, axis=1)
        cross_term = np.dot(X, self.X_train.T)
        distances = np.sqrt(X_square - 2 * cross_term + train_square)

        return distances

    def score(self, X, y):
        y_pred = self.predict(X)

        return np.mean(y_pred == y)
