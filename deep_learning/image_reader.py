# opencv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
file_name = '../ensemble/beauty.png'
img = cv2.imread(filename=file_name)
cv2.imshow(winname='img', mat=img)
cv2.waitKey(delay=1000)
print(img)
# [H, W, C] height, width, channels通道数：OpenCV通道排布：BGR模式，而Maplotlib: RGB模式
print(img.shape)

image = plt.imread(fname=file_name)
plt.imshow(image)
plt.show()

pil_imag = Image.open(file_name)
print(np.array(pil_imag))
print(pil_imag.size)
pil_imag.show()
