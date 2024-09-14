"""
    2，搭建一个可用于手写数字识别的卷积网络模型（黑白图像，28*28大小，10个分类；可借助 PyTorch）；
"""
import torch
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.mp1 = nn.MaxPool2d(kernel_size=2, 
                                stride=2, 
                                padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, 
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.mp2 = nn.MaxPool2d(kernel_size=2, 
                                        stride=2, 
                                        padding=0)
        self.linear1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.mp2(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.linear2(x)

        return x
