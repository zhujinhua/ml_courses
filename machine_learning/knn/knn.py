import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import logging

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from MyKNN import MyKNN
from matplotlib.pyplot import plot

logging.basicConfig(level=logging.INFO)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# logging.info(y_pred == y_test)
# logging.info((y_pred == y_test).mean())
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# y_dtc_pred = dtc.predict(X_test)
# logging.info(y_dtc_pred == y_test)
# logging.info((y_dtc_pred == y_test).mean())
# svc = SVC()
# svc.fit(X_train, y_train)
# svc_pred = svc.predict(X_test)
# logging.info(svc_pred == y_test)
# logging.info((svc_pred == y_test).mean())
# joblib.dump(value=svc, filename='svc_iris.joblib')
# load_svc = joblib.load(filename='svc_iris.joblib')
# y_svc_pred = load_svc.predict(X_test)
# logging.info(y_svc_pred == y_test)
# KNN Algorithm
# step 1: only load data into memory, it's a lazy algorithm; actually it doesn't learn, it is
#         based on rules and data
# step2: choose k, found the K nearest neighbours by Euclidean distance or 向量视角，用余玄相似度度量
# step3: voting based on k nearest neighbours, it will belong to which is the most
my_knn = MyKNN(n_neighbors=5, metric='minkowski')
my_knn.fit(X_train, y_train)
# joblib.dump(value=my_knn, filename='myKNN.joblib')
# save_knn = joblib.load(filename='myKNN.joblib')
my_knn_predict = my_knn.predict(X_test)
logging.info('My KNN: %s',  my_knn_predict == y_test)


