"""
Author: jhzhu
Date: 2024/6/20
Description: 
"""
import torch
from torch import nn

from deep_learning.utils.ConvBlock import ConvBlock


class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extraction = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            # sub sampling
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvBlock(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x)
        return x
