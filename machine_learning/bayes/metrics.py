import logging

import numpy as np

x = np.random.randint(low=20, high=101, size=50)
x.max()
x.mean()

X_1 = np.array([1, 3, 5, 7, 9])
Y_1 = np.array([2, 4, 6, 8, 10])
E = ((X_1 - X_1.mean())*(Y_1 - Y_1.mean())).mean()/X_1.std()/Y_1.std()

logging.error(f'{E}')
