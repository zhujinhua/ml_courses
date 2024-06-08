import numpy as np
from sklearn.datasets import load_iris
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from probability import gausssian_fun
from MyGaussian import MyGaussian
from OptimizeGaussian import OptimizeGaussian

logging.basicConfig(level=logging.INFO)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
logging.info(y_pred)

# my_gnb = MyGaussian()
# my_gnb.fit(X_train, y_train)
# my_pred = my_gnb.predict(X_test)
# logging.info(my_pred)

opt_gnb = OptimizeGaussian()
opt_gnb.fit(X_train, y_train)
opt_pred = opt_gnb.predict(X_test)
logging.info(opt_pred)
logging.info((y_pred == opt_pred).mean())


x5 = X_test[5]
y5 = y_test[5]

X0 = X_train[y_train == 0]
X1 = X_train[y_train == 1]
X2 = X_train[y_train == 2]


p0 = ((y_train == 0).mean() * gausssian_fun(x5[0], X0.mean(axis=0)[0], X0.std(axis=0)[0])
      * gausssian_fun(x5[1], X0.mean(axis=0)[1], X0.std(axis=0)[1])
      * gausssian_fun(x5[2], X0.mean(axis=0)[2], X0.std(axis=0)[2])
      * gausssian_fun(x5[3], X0.mean(axis=0)[3], X0.std(axis=0)[3]))
p1 = ((y_train == 1).mean() * gausssian_fun(x5[0], X1.mean(axis=0)[0], X1.std(axis=0)[0])
      * gausssian_fun(x5[1], X1.mean(axis=0)[1], X1.std(axis=0)[1])
      * gausssian_fun(x5[2], X1.mean(axis=0)[2], X1.std(axis=0)[2])
      * gausssian_fun(x5[3], X1.mean(axis=0)[3], X1.std(axis=0)[3]))
p2 = ((y_train == 2).mean() * gausssian_fun(x5[0], X2.mean(axis=0)[0], X2.std(axis=0)[0])
      * gausssian_fun(x5[1], X2.mean(axis=0)[1], X2.std(axis=0)[1])
      * gausssian_fun(x5[2], X2.mean(axis=0)[2], X2.std(axis=0)[2])
      * gausssian_fun(x5[3], X2.mean(axis=0)[3], X2.std(axis=0)[3]))

logging.info(f'{p0} {p1} {p2}')
logging.info(y_pred[5])
