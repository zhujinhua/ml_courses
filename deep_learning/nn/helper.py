import numpy as np
import matplotlib.pyplot as plt
import torch

#  classification: model only output the ratio of all the classed
logits = [0.6, -3.6, 18.9]


# sigmod(二分类), softmax（多分类模拟）

def sigmoid(x):
    # (0, 1)
    return 1 / (1 + np.exp(-x))


x = np.linspace(start=-100, stop=100, num=300)
plt.plot(sigmoid(x), c='red')


def softmax(logits):
    # increase function,
    # x = 0, y=0.5 ????
    # x<0, y<0.5,
    # x>0 y>0.5
    logits = np.array(logits)
    return np.exp(logits) / np.exp(logits).sum()


plt.plot(softmax(x), c='blue')

plt.grid()
plt.show()

# one-hot encoding:
# 互相垂直的向量，没有先验的远近关系
# 都是比较长的向量，跟类别数量一致
# 每个向量只有1位是1，其余都是0，高度稀疏sparse
# 从计算和存储上来说，都比较浪费

y_true = np.array([0, 1, 0])
y_pred1 = softmax([12.5, -0.5, 2.7])
y_pred2 = softmax([-12.5, 6.4, 2.7])
loss1 = np.log(softmax(-0.5))  #交叉熵计算代价比较小，只计算一个
loss2 = np.log(softmax(6.4))

logits = torch.randn(5, 3)
logits.argmax(dim=-1)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


x = np.linspace(start=-10, stop=10, num=100)
plt.plot(tanh(x))
plt.show()

x = torch.randn(3, 5)
x.relu()

x = np.linspace(start=-20, stop=20, num=50)


def relu(x):
    x = np.array(x)
    x[x < 0] = 0
    return x


plt.plot(relu(x))
plt.show()
