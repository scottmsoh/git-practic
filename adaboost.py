import numpy as np


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
			predictions[feature > self.threshold] = -1

		return predictions


class AdaBoost:
	def __init__(
		self,
		n_estimators: int = 10,
		epsilon: float = 1e-10,
	):
		self.n_estimators = n_estimators
		self.epsilon = epsilon
		self.models = []
	
	def fit(self, X, y):
		n_samples, n_features = X.shape
		weights = np.full(n_samples, (1/n_samples))

		for _ in range(self.n_estimators):
			stump = Stump()
			min_error = float('inf')

			for feature_idx in range(n_features):
				threshold_candidates = np.unique(X[:, feature_idx])
				for threshold in threshold_candidates:
					for polarity in (-1, 1):
						predictions = np.ones(n_samples)

						if polarity == 1:
							predictions[X[:, feature_idx] < threshold] = -1
						else:
							predictions[X[:, feature_idx] > threshold] = -1
						
						misclassified = weights[y != predictions]
						total_error = misclassified.sum()

						if total_error < min_error:
							stump.polarity = polarity
							stump.threshold = threshold
							stump.feature_index = feature_idx
							min_error = total_error

			stump.amount_of_say = 0.5*np.log((1-min_error+self.epsilon) / min_error+self.epsilon)
			predictions = stump.predict(X)
			weights *= np.exp(-stump.amount_of_say * y * predictions)
			weights /= np.sum(weights)

			self.models.append(stump)
	
	def predict(self, X):
		stump_predictions = [model.amount_of_say*model.predict(X) for model in self.models]
		y_pred = np.sign(np.sum(stump_predictions))

		return y_pred
