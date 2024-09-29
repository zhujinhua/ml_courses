"""
Author: jhzhu
Date: 2024/9/27
Description: 
"""
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HandWrittenNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = ConvBlock(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
