import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg


# threshold
img_src = cv2.imread('/Users/jamie/Desktop/opencv/Spider - Photo/pic-20191015132516(1-88).png',  cv2.IMREAD_GRAYSCALE) 
ret, img_dst = cv2.threshold(img_src, 115, 255, cv2.THRESH_BINARY)

white = []  # 记录每一列的白色像素总和
black = []  # 记录每一列的黑色像素总和
height = img_dst.shape[0]
width = img_dst.shape[1]
print(width, height)
white_max = 0   # 仅保存每列，取列中白色最多的像素总数
black_max = 0   # 仅保存每列，取列中黑色最多的像素总数

# 循环计算每一列的黑白色像素总和
for i in range(width):
    w_count = 0     # 这一列白色总数
    b_count = 0     # 这一列黑色总数
    for j in range(height):
        if img_dst[j][i] == 255:
            w_count += 1
        else:
            b_count += 1
    white_max = max(white_max, w_count)
    black_max = max(black_max, b_count)
    white.append(w_count)
    black.append(b_count)


# False表示白底黑字；True表示黑底白字
arg = black_max > white_max
print(arg)
