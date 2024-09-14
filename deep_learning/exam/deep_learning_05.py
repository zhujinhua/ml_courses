"""
    5，设计一个可用于计算机视觉任务的数据增强处理（可借助 torchvision）；
"""

from torchvision import transforms

# 数据增强
trans = transforms.Compose([
    transforms.ColorJitter(brightness=0.1,
                           contrast=0.1,
                           saturation=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10)
])
