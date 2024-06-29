"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""
import torch.nn as nn


class ConvBlock(nn.Module):
    """
        虚线块，是 y = F(x) + Conv(x)
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stage = nn.Sequential(
            # 三层卷积
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels[0],
                      kernel_size=1,
                      stride=stride,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(inplace=True),
            # 2
            nn.Conv2d(in_channels=out_channels[0],
                      out_channels=out_channels[1],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[1]),
            nn.ReLU(inplace=True),
            # 3
            nn.Conv2d(in_channels=out_channels[1],
                      out_channels=out_channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[2]))

        # 短路层
        self.shortcut = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels[2],
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        s = self.shortcut(x)
        s = self.bn(s)
        h = self.stage(x)
        h = h + s
        o = self.relu(h)
        return o


class IdentityBlock(nn.Module):
    """
        实线块，是 y = F(x) + x
    """
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.stage = nn.Sequential(
            # 1：1 x 1
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels[0],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(inplace=True),
            # 2：3 x 3
            nn.Conv2d(in_channels=out_channels[0],
                      out_channels=out_channels[1],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[1]),
            nn.ReLU(inplace=True),
            # 3：1 x 1
            nn.Conv2d(in_channels=out_channels[1],
                      out_channels=out_channels[2],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels[2])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        s = x
        h = self.stage(x)
        h = s + h
        o = self.relu(h)
        return o


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      padding=3,
                      stride=2,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(in_channels=64,
                      out_channels=(64, 64, 256),
                      stride=1),
            IdentityBlock(in_channels=256,
                          out_channels=(64, 64, 256)),
            IdentityBlock(in_channels=256,
                          out_channels=(64, 64, 256)),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(in_channels=256,
                      out_channels=(128, 128, 512),
                      stride=2),
            IdentityBlock(in_channels=512,
                          out_channels=(128, 128, 512)),
            IdentityBlock(in_channels=512,
                          out_channels=(128, 128, 512)),
            IdentityBlock(in_channels=512,
                          out_channels=(128, 128, 512))
        )

        self.stage4 = nn.Sequential(
            ConvBlock(in_channels=512,
                      out_channels=(256, 256, 1024),
                      stride=2),
            IdentityBlock(in_channels=1024,
                          out_channels=(256, 256, 1024)),
            IdentityBlock(in_channels=1024,
                          out_channels=(256, 256, 1024)),
            IdentityBlock(in_channels=1024,
                          out_channels=(256, 256, 1024)),
            IdentityBlock(in_channels=1024,
                          out_channels=(256, 256, 1024)),
            IdentityBlock(in_channels=1024,
                          out_channels=(256, 256, 1024))
        )
        self.stage5 = nn.Sequential(
            ConvBlock(in_channels=1024,
                      out_channels=(512, 512, 2048),
                      stride=2),
            IdentityBlock(in_channels=2048,
                          out_channels=(512, 512, 2048)),
            IdentityBlock(in_channels=2048,
                          out_channels=(512, 512, 2048))
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048,
                            out_features=10)

    def forward(self, x):
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.stage5(h)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        o = self.fc(h)
        return o
