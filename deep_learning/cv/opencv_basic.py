import cv2
import numpy as np

file_name = '../../dataset/beauty.png'
img = cv2.imread(filename=file_name)
# 简单滤波处理,平滑滤波=卷积操作
N_1 = 3
kernel_1 = np.ones((N_1, N_1)) / N_1 ** 2  # 滤波????

N_2 = 7
kernel_2 = np.ones((N_2, N_2)) / N_2 ** 2  # 滤波????

N_3 = 11
kernel_3 = np.ones((N_3, N_3)) / N_3 ** 2  # 滤波????
# 目标图像的深度？？？，RGB=3，
img1 = cv2.filter2D(img, ddepth=-1, kernel=kernel_1)  # 对信号的所有处理均是滤波？？？？
img2 = cv2.filter2D(img, ddepth=-1, kernel=kernel_2)  # 对信号的所有处理均是滤波？？？？
img3 = cv2.filter2D(img, ddepth=-1, kernel=kernel_3)  # 对信号的所有处理均是滤波？？？？

cv2.imshow(winname='raw image', mat=img)
cv2.waitKey(delay=2000)

cv2.imshow(winname='filtered 1 image', mat=img1)
cv2.waitKey(delay=2000)

cv2.imshow(winname='filtered 2 image', mat=img2)
cv2.waitKey(delay=2000)

cv2.imshow(winname='filtered 3 image', mat=img3)
cv2.waitKey(delay=2000)

N = 3
# 抽取第三行与第一行的差别
kernel_4 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
img4 = cv2.filter2D(img, ddepth=-1, kernel=kernel_4)
img5 = cv2.filter2D(img, ddepth=-1, kernel=kernel_4.T)
cv2.imshow(winname='filtered 4 image', mat=img4)
cv2.waitKey(delay=2000)
cv2.imshow(winname='filtered 5 image', mat=img5)
cv2.waitKey(delay=2000)

kernel_6 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 减上下左右，锐化
img6 = cv2.filter2D(img, ddepth=-1, kernel=kernel_6)
cv2.imshow(winname='filtered 6 image', mat=img6)
cv2.waitKey(delay=3000)

# 不同的算子，设计kernel，对图片做滤波