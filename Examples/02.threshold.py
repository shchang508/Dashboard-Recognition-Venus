import cv2
import numpy as np

# read image
img_src = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# threshold
ret, img_dst1 = cv2.threshold(img_src, 100, 255, cv2.THRESH_BINARY)
ret, img_dst2 = cv2.threshold(img_src, 220, 255, cv2.THRESH_BINARY)

# show image
cv2.imshow('img_src', img_src)
cv2.imshow('img_dst1', img_dst1)
cv2.imshow('img_dst2', img_dst2)

# waitKey
key = cv2.waitKey(0)
