import numpy as np


class MyGaussian(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        '''
        :param X:
        :param y:
        :return:
        '''
        self.X = X
        self.y = y
        # get the number of classifier
        self.classifier_list = np.unique(self.y).tolist()
        self.feature_number = self.X.shape[1]

    def predict(self, X):
        '''
        :param X:
        :return:
        '''
        X = np.array(X)
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError('Parameter has something wrong, please check!')
        results = []
        for x in X:
            p_max = 0
            x_classifier = None
            for c in self.classifier_list:
                mu = self.X[self.y == c].mean(axis=0)
                sigma = self.X[self.y == c].std(axis=0)
                p = ((self.y == c).mean())
                for i in range(self.feature_number):
                    p = p * (self.__gausssian(x[i], mu[i], sigma[i]))
                if p > p_max:
                    p_max = p
                    x_classifier = c
            results.append(x_classifier)
        return np.array(results)

    def __gausssian(self, x, mu, sigma):
        '''
        :param x:
        :param mu:
        :param sigma:
        :return:
        '''
        return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
