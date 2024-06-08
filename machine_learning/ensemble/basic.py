import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
# 检查是否有可用的GPU
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型和数据
model = SimpleModel().to(device)
data = torch.randn(64, 784).to(device)  # 64个样本，每个样本有784个特征
labels = torch.randint(0, 10, (64,)).to(device)  # 64个标签，值在0到9之间

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()  # 清零梯度
    outputs = model(data)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    logging.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# array_test = np.arange(12)
# array_reshape = np.random.randint(5, 30, 12).reshape(3, 4)
#
# reshape_test = array_test.reshape(3, 4)
# logging.info(reshape_test)
#
# result = np.stack(arrays=[array_reshape, reshape_test], axis=0)
# con = np.concatenate([array_reshape, reshape_test], axis=-1)
# logging.info(result)
# logging.info(con)
#
# scores = np.random.randint(0, 101, 50)
# plt.figure()
# plt.plot(scores - scores.mean())
# plt.show()
# # normalization
# logging.info((scores - scores.min()) / (scores.max() - scores.min()))
#
# # standardize
# plt.plot((scores - scores.mean()) / scores.std())
#
# # pytorch: used as compute structure
# # tensor: array
# tensor = torch.tensor(data=[1, 2, 3])
# torch.ones(6)
# torch.rand(2, 3)
# torch.randn(2, 3)
# torch.randint(low=0, high=101, size=(2, 3))
# torch.linspace(start=-5, end=50, steps=50)
# t = torch.randn(3, 5)
# logging.info(torch.cuda.is_available())
