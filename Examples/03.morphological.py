import cv2
import numpy as np

# read image
img_src = cv2.imread('/Users/jamie/Desktop/opencv/speedPositive/pic-20190924142548(1-15).png', cv2.IMREAD_GRAYSCALE)

# threshold
ret, img_dst = cv2.threshold(img_src, 115, 255, cv2.THRESH_BINARY)
# img_dst1 = cv2.adaptiveThreshold(img_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# img_dst2 = cv2.adaptiveThreshold(img_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# morphological
kernel = np.ones((2, 2), np.uint8)
img_dst1 = cv2.erode(img_dst, kernel, iterations = 1)
img_dst2 = cv2.dilate(img_dst, kernel, iterations = 1)

# show image
cv2.imshow('img_dst', img_dst)
cv2.imshow('img_dst1', img_dst1)
cv2.imshow('img_dst2', img_dst2)

# waitKey
key = cv2.waitKey(0)
