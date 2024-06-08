import numpy as np
from collections import Counter


class MyKNN(object):
    def __init__(self, n_neighbors, metric="minkowski"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError('Parameter has something wrong, please check!')
        self.X = X
        self.y = y

    def predict(self, X):
        X = np.array(X)
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError('Parameter has something wrong, please check!')
        results = []
        for x in X:
            if self.metric == 'euclidean':
                distance = self.__euclid_distance(x)
                index = np.argsort(distance)[: self.n_neighbors]
            elif self.metric == 'minkowski':
                distance = self.__minkowski_distance(x)
                index = np.argsort(distance)[: self.n_neighbors]
            elif self.metric == 'cityblock':
                distance = self.__manhattan_distance(x)
                index = np.argsort(distance)[: self.n_neighbors]
            elif self.metric == 'cosine':
                distance = self.__cosine_similarity(x)
                index = np.argsort(distance)[-self.n_neighbors:]
            label = Counter(self.y[index]).most_common(1)[0][0]
            results.append(label)
        return np.array(results)

    def __euclid_distance(self, x):
        return np.sqrt(np.sum((self.X - x) ** 2, axis=-1))

    def __manhattan_distance(self, x):
        return np.sum(np.abs(self.X - x), axis=-1)

    def __cosine_similarity(self, x):
        norm_X = np.linalg.norm(self.X, axis=1)
        norm_x = np.linalg.norm(x)
        return np.dot(self.X, x) / (norm_X * norm_x)

    def __minkowski_distance(self, x, p=3):
        return np.sum((np.abs(self.X - x))**p, axis=-1)**(1 / p)
