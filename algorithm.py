import numpy as np


class DecisionTreeNode:
	def __init__(
		self,
		feature_index=None,
		threshold=None,
		left=None,
		right=None,
		*,
		value=None,
		):
		self.feature_index = feature_index
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def is_leaf_node(self):
		return self.value is not None


class DecisionTree:
	def __init__(self, max_depth=5, min_samples_split=2):
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.root = None

	def fit(self, X, y):
		self.root = self._grow_tree(X, y)

	def _grow_tree(self, X, y, depth=0):
		num_samples, num_features = X.shape
		num_labels = len(np.unique(y))

	    if (depth  self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
			leaf_value = self._most_common_label(y)
			return DecisionTreeNode(value=leaf_value)

		best_feature, best_threshold = self._find_best_split(X, y, num_features)

		if best_feature is None:
			return DecisionTreeNode(value=self._most_common_label(y))

		left_indices = X[:, best_feature] <= best_threshold
		right_indices = X[:, best_feature] > best_threshold

		left_tree = self._grow_tree(X[left_indices], y[left_indices], depth+1)
		right_tree = self._grow_tree(X[right_indices], y[right_indices], depth+1)

		return DecisionTreeNode(best_feature, best_threshold, left_tree, right_tree)

	def _find_best_split(self, X, y, num_features):
		best_gain = -1
		split_idx, split_threshold = None, None

		for feature_index in range(num_features):
			thresholds = np.unique(X[:, feature_index])
			for threshold in thresholds:
				left = y[X[:, feature_index] <= threshold]
				right = y[X[:, feature_index] threshold]

				if len(left) == 0 or len(right) == 0:
					continue

				information_gain = self._information_gain(y, left, right)

				if information_gain best_gain:
					best_gain = information_gain
					split_idx = feature_index
					split_threshold = threshold

		return split_idx, split_threshold

	# def _find_best_split(self, X, y, num_features):
	# 	best_gain = -1
	# 	split_idx, split_thresh = None, None

	# 	for feature_index in range(num_features):
	# 		sorted_indices = X[:, feature_index].argsort()
	# 		X_sorted = X[sorted_indices]
	# 		y_sorted = y[sorted_indices]

	# 		for i in range(1, len(X_sorted)):
	# 			if y_sorted[i] == y_sorted[i-1]:
	# 				continue

	# 			threshold = (X_sorted[i, feature_index] + X_sorted[i-1, feature_index]) / 2

	# 			left = y_sorted[:i]
	# 			right = y_sorted[i:]

	# 			information_gain = self._information_gain(y_sorted, left, right)

	# 			if information_gain best_gain:
	# 				best_gain = information_gain
	# 				split_idx = feature_index
	# 				split_thresh = threshold

	# 	return split_idx, split_thresh

	def _entropy(self, y):
		hist = np.bincount(y)
		ps = hist / len(y)

		return -np.sum([p*np.log2(p) for p in ps if p 0])

	def _information_gain(self, parent, left, right):
		weight_left = len(left) / len(parent)
		weight_right = len(right) / len(parent)
		information_gain = self._entropy(parent) - (weight_left*self._entropy(left) + weight_right*self._entropy(right))

		return information_gain

	def _most_common_label(self, y):
		counts = np.bincount(y)

		return np.argmax(counts)

	def predict(self, X):
		return np.array([self._traverse_tree(x, self.root) for x in X])

	def _traverse_tree(self, x, node):
		if node.is_leaf_node():
			return node.value

		if x[node.feature_index] <= node.threshold:
			return self._traverse_tree(x, node.left)

		return self._traverse_tree(x, node.right)
	
