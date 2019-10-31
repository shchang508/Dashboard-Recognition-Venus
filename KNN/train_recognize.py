import cv2, os
import shutil
import numpy as np
from PIL import Image, ImageChops
from time import sleep
import sys

# set recognized area
'''x_min1 = 600
x_max1 = 950
y_min1 = 420
y_max1 = 620
x_min2 = 0
x_max2 = 1279
y_min2 = 750
y_max2 = 800'''

# chop image
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 4.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# sort contours
def get_contour_precedence(contour, cols):
    # tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return cols + origin[0]

# return result
def get_reult():
    return num_data

# train model
samples = np.loadtxt('samples.data', np.float32)
responses = np.loadtxt('responses.data', np.float32)
print('Sample path:', os.path.abspath('./KNN/samples.data'))
print('Respons path:', os.path.abspath('./KNN/response.data'))
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# read image
# img_src = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img_src = cv2.imread('D:/Dashboard-Recognition-Venus/SpeedPhoto_all/pic-20191015131920(1-29).png', cv2.IMREAD_GRAYSCALE)
#img_src = img_src[350:950, 300:1700]
#print(img_src.shape)
img_mean = np.mean(img_src)

# threshold
if img_mean >= 50: 
	print('White bakground')
	img_src = trim(Image.fromarray(img_src))
	ret, img_dst = cv2.threshold(np.array(img_src), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)	
else:
	print('Black background')
	ret, img_dst = cv2.threshold(img_src, 200, 255, cv2.THRESH_BINARY)

# morphological
kernel = np.ones((3, 3), np.uint8)
img_dst = cv2.erode(img_dst, kernel, iterations = 1)

# set output image
img_out = img_dst.copy()
img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)

# set recognized area
'''cv2.rectangle(img_out, (x_min1, y_min1), (x_max1, y_max1), (0, 0, 255), 1)
cv2.line(img_out, (x_min2, y_min2), (x_max2, y_min2), (0, 0, 255), 1)
cv2.line(img_out, (x_min2, y_max2), (x_max2, y_max2), (0, 0, 255), 1)'''

# train each contour 
contours, _ = cv2.findContours(img_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours.sort(key=lambda x:get_contour_precedence(x, img_dst.shape[1]))


idx = 0
num_str = []
for contour in contours:
	if cv2.contourArea(contour) > 2000:
		[x, y, w, h] = cv2.boundingRect(contour)

		if (x > 0) and (y > 0) and (h > 150) and (w < 300):
			cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 255, 0), 2)
			print('x:', x, ' y:', y, ' w:', w, ' h:', h)
			cv2.imshow('img_out', img_out)

			img_roi = img_dst[y:y+h, x:x+w]
			img_roi = cv2.resize(img_roi, (20 , 20))
			cv2.imshow('img_roi', img_roi)

			img_roi = img_roi.reshape((1, 400))
			img_roi = np.float32(img_roi)
				
			# recognize number
			retval, results, neigh_resp, dists = model.findNearest(img_roi, k=1)
			num = str(int((results[0][0])))
			print('Num: ', num)
			cv2.putText(img_out, num, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
			
			cv2.imshow('img_out', img_out)
			num_str.append(num)
			# idx += 1

idx += 1
key = cv2.waitKey(0)
    
# write image
cv2.imwrite('output.png', img_out, [cv2.IMWRITE_JPEG_QUALITY, 90])
number = map(int, num_str)
temp = []
for i in num_str:
	temp.append(i)
# temp.reverse()
temp_str = ''
num_data = temp_str.join(temp)
print('Result:', num_data)
get_reult()

# SaveDirectory = os.getcwd()
path = "D:\\OpenCV_Result"
if os.path.isdir(path):  # delete directory if it exists
    # shutil.rmtree(path)
    sleep(2)  # delay to avoid error
else:
    os.mkdir(path)

# name = os.path.join(path,'result_' + time.strftime('%Y%m%d%H%M%S') + '.txt')
file = open('result.txt', 'a')
file.write(num_data + ';')
file.close()