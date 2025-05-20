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

class Stump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.amount_of_say = None

    def predict(self, X):
        n_samples = len(X)
        predictions = np.ones(n_samples)
        feature = X[:, self.feature_index]

        if self.polarity == 1:
            predictions[feature < self.threshold] = -1
        else:
            predictions[feature  self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_estimators=10, epsilon=1e-10):
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.full(n_samples, (1/n_samples))

       for _ in range(self.n_estimators):
           stump = Stump()
           min_error = float('inf')
           for feature_i in range(n_features):
               feature_values = np.unique(X[:, feature_i])
               for threshold in feature_values:
                   for polarity in [1, -1]:
                       predictions = np.ones(n_samples)
                       if polarity == 1:
                           predictions[X[:, feature_i] < threshold] = -1
                       else:
                           predictions[X[:, feature_i]  threshold] = -1
                       misclassified = weights[y != predictions]
                       error = misclassified.sum()
                       if error < min_error:
                           stump.polarity = polarity
                           stump.threshold = threshold
                           stump.feature_index = feature_i
                           min_error = error
           stump.amount_of_say = 0.5*np.log((1-min_error+self.epsilon) / (min_error+self.epsilon))
           predictions = stump.predict(X)
           weights *= np.exp(-stump.amount_of_say * y * predictions)
           weights /= np.sum(weights)
           self.models.append(stump)
   def predict(self, X):
       stump_predictions = [model.alpha*model.predict(X) for model in self.models]
       y_pred = np.sign(np.sum(stump_predictions, axis=0))

       return y_pred


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)

        return X[indices]

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis]-self.centroids, axis=2)

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            if np.any(labels == i):
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = X[np.random.choice(X.shape[0])]

        return new_centroids

    def _is_converged(self, old_centroids, new_centroids):
        shift = np.linalg.norm(old_centroids-new_centroids, axis=1)

        return np.all(shift < self.tol)

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)

            if self._is_converged(self.centroids, new_centroids):
                print(f'Converged at iteration {i}')
                break

            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(self._compute_distances(X))

    def predict(self, X):
        assert self.centroids is not None, 'Model has not been fitted yet.'

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


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
