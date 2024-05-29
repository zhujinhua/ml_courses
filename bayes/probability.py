import numpy as np
from sklearn.datasets import load_iris
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
X, y = load_iris(return_X_y=True)
p_y0 = (y == 0).mean()
p_y1 = (y == 1).mean()
p_y2 = (y == 2).mean()

logging.info(f'{p_y0} {p_y1} {p_y2}')


def gausssian_fun(x, mu, sigma):
    '''
    :param x:
    :param mu:
    :param sigma:
    :return:
    '''
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


x = np.linspace(-5, 5, num=50)
plt.plot(x, gausssian_fun(x, 0, 1), label=r'$\mu=0, \sigma=1$')
plt.plot(x, gausssian_fun(x, 1, 1), label=r'$\mu=1, \sigma=1$')
plt.plot(x, gausssian_fun(x, 0, 2), label=r'$\mu=0, \sigma=2$')
plt.grid()
plt.legend()
plt.show()

logits = np.array([1, 2, 2, 2, 2, 3, 1, 1])
sigma = logits.std()
mu = logits.mean()

p_1 = gausssian_fun(x=1, mu=mu, sigma=sigma)
p_2 = gausssian_fun(x=2, mu=mu, sigma=sigma)
p_3 = gausssian_fun(x=3, mu=mu, sigma=sigma)

logging.info(f'{p_1}, {p_2}, {p_3}')
