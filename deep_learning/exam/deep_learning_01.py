"""
    1，搭建一个可用于波士顿房价预测的全连接网络模型（可借助 PyTorch）；
"""
import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, n_features=13):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features=n_features, out_features=8)
        self.linear2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
