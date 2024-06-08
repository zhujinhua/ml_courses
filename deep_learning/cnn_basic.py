from torch import nn

# 输入图像的通道数，输出图像的通道数，卷积核的尺寸（神经网络自己设计卷积核，人工智能核是学习参数，只需要设置尺寸大小 比如3*3）
# stride 是每次移动步长，默认是1；padding 图像填充,上下左右对称填充 (kernel_size - 1)/2,默认不填充
# 细枝末节，花活：dilation 膨胀/空洞卷积， groups 分组卷积
# bias 偏置 相乘相加再加个值
# padding_mode:填充方式，填什么都一样，但0比较简单,计算比较快
# device 后期处理
# dtype torch float 32
conv = nn.Conv2d(in_channels=3,
          out_channels=8,  # 每一层特征图???
          kernel_size=3,
          stride=1,
          padding=1)
# 卷积层的参数是核
weights = conv.weight.shape
bias = conv.bias.shape
