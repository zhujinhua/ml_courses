# y = w0*x0 + w1*x1 +。。。+ w12*x12 + b
# y = x@w + b
# y = X@w + b: X feature matrix, calculate w, b
import torch
from torch import nn # layer

# 全连接， 感知机，稠密层，线性层， 俄罗斯人写的矩阵论
linear_layer = nn.Linear(in_features=13, out_features=1)

print('Weights: %s' % linear_layer.weight)
print('Bias: %s' % linear_layer.bias)
# Mean Squared Error
loss = nn.MSELoss()
# 优化过程：求梯度的过程
optimizer = torch.optim.SGD(params=linear_layer.parameters(), lr=1e-3) # random gradient descent: 减梯度，清空梯度
# 清空梯度
optimizer.zero_grad()
# 减去偏导（计算偏导）
loss.backward()
# 更新参数
optimizer.step()
