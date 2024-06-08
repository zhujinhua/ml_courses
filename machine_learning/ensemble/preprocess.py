from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# centralization: minus mean value
bonston_pd = pd.read_csv('../../dataset/BostonHousing.csv')
X = bonston_pd.iloc[:, :-1]
y = bonston_pd.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
# MSE 凸函数，求导比较方便
logging.info(f'KNN MSE: {mean_squared_error(y_test, knn_pred)}')
logging.info(f'KNN MAE: {mean_absolute_error(y_test, knn_pred)}')

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

logging.info(f'Decision Tree MSE: {mean_squared_error(y_test, tree_pred)}')
logging.info(f'Decision Tree MAE: {mean_absolute_error(y_test, tree_pred)}')
# 权重，偏置
linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

logging.info(f'Linear Model MSE: {mean_squared_error(y_test, linear_pred)}')
logging.info(f'Linear Model MAE: {mean_absolute_error(y_test, linear_pred)}')
logging.info(f'{linear.coef_} {linear.intercept_}')

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegression(max_iter=10000)  # check max iter
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
logging.info((lr_pred == y_test).mean())

pipeline = Pipeline([
    ('Standard', StandardScaler()),
    ('PCA', PCA(n_components=8)),
    ('estimator', LinearRegression())
])
pipeline.fit(X_train, y_train)
p_pred = pipeline.predict(X_test)
logging.info(f'linear regression MAE: {mean_absolute_error(y_true=y_test, y_pred=p_pred)}')
logging.info(f'linear regression MSE: {mean_squared_error(y_true=y_test, y_pred=p_pred)}')
logging.info(f'linear regression R2: {r2_score(y_true=y_test, y_pred=p_pred)}')

# Load and display the image
beauty = plt.imread(fname='beauty.png')
plt.imshow(beauty)
plt.show()

# Perform SVD on the red channel
r_beauty = beauty[:, :, 0]
U, S, Vt = np.linalg.svd(a=r_beauty, full_matrices=False)

# Plot the singular values
plt.plot(S)
plt.title("Singular Values of the Red Channel")
plt.show()

# Logging the reconstruction using all singular values
reconstructed_all = U @ np.diag(S) @ Vt
logging.info("Reconstructed (all singular values): {}".format(reconstructed_all))

# Logging the low-rank approximation using the first 5 singular values
k = 100
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
reconstructed_k = U_k @ S_k @ Vt_k
logging.info("Reconstructed (first 5 singular values): {}".format(reconstructed_k))

# If needed, display the low-rank approximation
plt.imshow(reconstructed_k, cmap='gray')
plt.title("Low-Rank Approximation (k=5)")
plt.show()
