import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch, channel, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# MBConv block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                                groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.se = SEBlock(mid_channels, reduction=int(1 / se_ratio))
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dwconv(x)))
        x = self.se(x)
        x = self.bn3(self.conv2(x))

        if self.skip:
            x += residual
        return x


# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, num_classes=1000, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        base_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList([])
        self._add_blocks(1, base_channels, base_channels * 6, 1, 1, 3)
        self._add_blocks(2, base_channels * 6, base_channels * 6, 2, 6, 3)
        self._add_blocks(2, base_channels * 6, base_channels * 6, 2, 6, 5)
        self._add_blocks(3, base_channels * 6, base_channels * 6, 1, 6, 3)
        self._add_blocks(3, base_channels * 6, base_channels * 6, 2, 6, 5)
        self._add_blocks(4, base_channels * 6, base_channels * 6, 1, 6, 3)
        self._add_blocks(1, base_channels * 6, 1280, 1, 6, 1)

        self.head = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def _add_blocks(self, num_blocks, in_channels, out_channels, stride, expand_ratio, kernel_size):
        for i in range(num_blocks):
            self.blocks.append(
                MBConv(in_channels, out_channels, expand_ratio, stride if i == 0 else 1)
            )
            in_channels = out_channels

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=num_classes)

# Example usage:
# model = efficientnet_b0(num_classes=1000)
# print(model)
