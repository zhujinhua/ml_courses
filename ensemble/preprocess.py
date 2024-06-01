from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
logging.basicConfig(level=logging.INFO)

# centralization: minus mean value
bonston_pd = pd.read_csv('BostonHousing.csv')
X = bonston_pd.iloc[:, :-1]
y = bonston_pd.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
# MSE 凸函数，求导比较方便
logging.info(mean_squared_error(y_test, knn_pred))
logging.info(mean_absolute_error(y_test, knn_pred))

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

logging.info(mean_squared_error(y_test, tree_pred))
logging.info(mean_absolute_error(y_test, tree_pred))