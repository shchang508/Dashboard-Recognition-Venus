import cv2
import numpy as np

# read image
img_src1 = cv2.imread('input.jpg', cv2.IMREAD_COLOR)
img_src2 = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# show image
cv2.imshow('img_src1', img_src1)
cv2.imshow('img_src2', img_src2)

# waitKey
key = cv2.waitKey(0)
