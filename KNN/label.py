import cv2, os
import numpy as np
from PIL import Image, ImageChops


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
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# read image
PATH = 'D:/Dashboard-Recognition-Venus/SpeedPhoto_all/'
imgs = os.listdir(PATH)
print(imgs)

# check if responses.data and samplps.data exist or not 
if os.path.isfile('responses.data'):
	responses = list(np.loadtxt('responses.data', dtype=int))
else:
	responses = list()

if os.path.isfile('samples.data'):
	samples = np.loadtxt('samples.data')
else:
	samples = np.empty((0,400))

# image processing
# for img in imgs[182:200]:
for img in imgs:
	img_path = PATH + img
    
    #img_src = cv2.imread('C:/Users/jamieshchang/Desktop/speedPositive/pic-20190924142455(1-2).png', cv2.IMREAD_GRAYSCALE)
	img_src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	print('Image: ', img_path)
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
	# img_dst1 = cv2.adaptiveThreshold(img_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

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
	_, contours, _ = cv2.findContours(img_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		if cv2.contourArea(contour) > 2000:
			[x, y, w, h] = cv2.boundingRect(contour)
			print(w,h)
			if (h > 150) and w < 300:
				cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 255, 0), 2)
				cv2.imshow('img_out', img_out)

				img_roi = img_dst[y:y+h, x:x+w]
				img_roi = cv2.resize(img_roi, (20 , 20))
				cv2.imshow('img_roi', img_roi)

				key = cv2.waitKey(0)

				if chr(key).isdigit():
					img_roi = img_roi.reshape((1, 400))
					responses.append(int(chr(key)))
					print('~~~~~~~~Key input~~~~~~~~', chr(key))
					samples = np.append(samples, img_roi, 0)
	print('~~~~~~~~Append key input~~~~~~~~', responses)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))

# output training data
np.savetxt('samples.data', samples)
np.savetxt('responses.data', responses, )
