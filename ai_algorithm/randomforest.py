from decision_tree import DecisionTree

import numpy as np
import random


class RandomForest:
	def __init__(
		self,
		n_estimators=10,
		max_depth=5,
		min_samples_leaf=2,
		max_features='sqrt',
		random_state=None,
		):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.trees = []
		self.random_state = random_state

		if random_state is not None:
			np.random.seed(random_state)
			random.seed(random_state)

	def _bootstrap_sample(self, X, y):
		n_samples = len(X)
		selected_indices = np.random.choice(n_samples, size=n_samples, replace=True)

		return X[selected_indices], y[selected_indices]

	def _get_feature_indices(self, n_features):
		if self.max_features == 'sqrt':
			return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
		elif isinstance(self.max_features, int):
			return np.random.choice(n_features, self.max_features, replace=False)
		else:
			return np.arange(n_features)

	def fit(self, X, y):
		self.trees = []
		n_features = X.shape[1]

		for _ in range(self.n_estimators):
			tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
			X_sample, y_sample = self._bootstrap_sample(X, y)
			feature_indices = self._get_feature_indices(n_features)
			tree.feature_indices = feature_indices
			tree.fit(X_sample[:, feature_indices], y_sample)
			self.trees.append((tree, feature_indices))

	def predict(self, X):
		predicted_labels_by_trees = np.array([
			tree.predict(X[:, feature_indices])
			for tree, feature_indices in self.trees
		])
		predicted_labels_by_trees = predicted_labels_by_trees.T

		return np.array([np.bincount(row).argmax() for row in predicted_labels_by_trees])
