"""
Author: jhzhu
Date: 2024/7/13
Description: 
"""
import time

from PIL import Image
from torchvision.transforms import transforms

# [w, h]
img = Image.open(fp='../../dataset/beauty.png')
img.show()
resize = transforms.Resize(size=(100, 100))
img1 = resize(img)
img1.show()
center_crop = transforms.CenterCrop(size=(2000, 100))  # RandomCrop
img2 = center_crop(img)
img2.show()
# HSI/HSV 色调饱和度亮度
color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
img3 = color_jitter(img)
img3.show()
compose = transforms.Compose(transforms=[color_jitter,
                                         center_crop,
                                         resize,
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])])
img4 = compose(img)
# img4.show()
flip = transforms.RandomVerticalFlip(p=1)  # RandomRotation
img5 = flip(img)
img5.show()


class MyRandomCrop(transforms.RandomCrop):
    def __int__(self):
        pass

    def forward(self, img):
        pass
