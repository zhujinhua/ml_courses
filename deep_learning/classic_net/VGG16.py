"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""
import torch.nn as nn
from deep_learning.utils.ConvBlock import ConvBlock


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


