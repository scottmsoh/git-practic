import numpy as np


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
 
 		n_samples, _ = X.shape
 		centroids = np.zeros((self.n_clusters, X.shape[1]))
 
 		idx = np.random.choice(n_samples)
 		centroids[0] = X[idx]
 
 
 		for i in range(1, self.n_clusters):
 			dist_sq = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2)**2, axis=1)
 
 			probabilities = dist_sq / dist_sq.sum()
 			next_idx = np.random.choice(n_samples, p=probabilities)
 			centroids[i] = X[next_idx]
 
 		return centroids
 
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
 ```