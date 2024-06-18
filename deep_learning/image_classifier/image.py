import torch
from torch import nn

'''
[N, C, H, W] 通道含义？？
 in_channels: int, 输入通道数
 out_channels: int, 输出通道数
 kernel_size: int | tuple[int, int],
 stride: int | tuple[int, int] = 1,  步长
 padding: str | int | tuple[int, int] = 0, 填充
 dilation: int | tuple[int, int] = 1, 膨胀系数，空洞卷积
 groups: int = 1,分组卷积
 bias: bool = True,偏置
 padding_mode: str = 'zeros',填充模式
 device: Any = None,设备 GPU, CPU
 dtype: Any = Non default torch.float 32
'''
conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
X = torch.randn(1, 3, 64, 64)
print(conv1(X).shape)

bn = nn.BatchNorm2d(num_features=16)


def relu(x):
    x[x < 0] = 0
    return x


X = torch.randn(2, 6)
print(relu(X))  #存在relu 变体


class ConvBlock(nn.Module):
    '''
    一层卷积： 卷积层，批规范化层 激活层
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.Relu(x)
        return x


con_block1 = ConvBlock(in_channels=3, out_channels=16)
X = torch.rand(1, 3, 64, 64)
con_block1(X)

# 线性层
linear = nn.Linear(in_features=1024, out_features=16)

#丢弃层，期望??
dp = nn.Dropout(p=0.5)
X = torch.rand(10)
print(dp(X))
dp.eval()
print(dp(X))
dp.train()
dp(X)
