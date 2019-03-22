from cluster import cluster
from copy import deepcopy
import numpy as np

class KMeans(cluster):
    def __init__(self, k= 5, max_iterations= 100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, Xin):
        X = np.mat(Xin)
        min = np.amin(Xin)
        max = np.amax(Xin)
        centroids = [
            [np.random.randint(min, max), np.random.randint(min, max)]
            for i in range(self.k)
        ]
        hypotheses = np.ones(len(X))
        for iteration in range(10):
            closest_centroids = deepcopy(hypotheses)
            hypotheses_old = deepcopy(hypotheses)
            for idx in range(len(X)):
                closest = np.ones(self.k)
                for centroid in range(len(centroids)):
                    closest[centroid] = np.linalg.norm(X[idx] - centroids[centroid])
                hypotheses[idx] = np.argmin(closest, axis=None)
                C = deepcopy(centroids)
                for i in range(self.k):
                    points = [X[j] for j in range(len(X)) if hypotheses[j] == i]
                    centroids[i] = np.mean(points, axis=0)
            if np.array_equal(hypotheses_old, hypotheses):
                break
        return hypotheses, centroids
