"""
Author: jhzhu
Date: 2024/5/30
Description: 
"""
import numpy as np


class OptimizeGaussian(object):
    def __init__(self):
        self.class_means = {}
        self.class_stds = {}
        self.class_priors = {}

    def fit(self, X, y):
        '''
        :param X: Training data
        :param y: Class labels
        '''
        self.X = X
        self.y = y
        self.classifier_list = np.unique(y)
        self.feature_number = X.shape[1]

        for c in self.classifier_list:
            X_c = X[y == c]
            self.class_means[c] = X_c.mean(axis=0)
            self.class_stds[c] = X_c.std(axis=0)
            self.class_priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        '''
        :param X: Data to predict
        :return: Predicted class labels
        '''
        X = np.array(X)
        if X.ndim != 2 or X.shape[1] != self.feature_number:
            raise ValueError('Parameter has something wrong, please check!')

        results = []
        for x in X:
            probabilities = {}
            for c in self.classifier_list:
                mu = self.class_means[c]
                sigma = self.class_stds[c]
                p = self.class_priors[c]
                p *= np.prod(self.__gaussian(x, mu, sigma))
                probabilities[c] = p

            x_classifier = max(probabilities, key=probabilities.get)
            results.append(x_classifier)
        return np.array(results)

    def __gaussian(self, x, mu, sigma):
        '''
        :param x: Feature value
        :param mu: Mean
        :param sigma: Standard deviation
        :return: Gaussian probability
        '''
        coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
        exponent = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        return coeff * exponent
