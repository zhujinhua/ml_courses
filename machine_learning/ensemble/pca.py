"""
Author: jhzhu
Date: 2024/6/3
Description:
标准化数据：将数据标准化，使其均值为0，方差为1。
计算协方差矩阵：对标准化数据计算协方差矩阵。
特征值分解：对协方差矩阵进行特征值分解，得到特征值和特征向量。
选择主成分：选择特征值最大的前k个特征向量作为主成分。
转换数据：将原始数据投影到主成分构成的新空间中。
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 标准化数据
X_centered = X - np.mean(X, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_centered, rowvar=False)

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 选择前两个主成分
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
eigenvectors = eigenvectors[:, :2]

# 转换数据
X_reduced = np.dot(X_centered, eigenvectors)

# 可视化
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()

