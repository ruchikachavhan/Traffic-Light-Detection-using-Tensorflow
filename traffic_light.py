import os 
import cv2
import numpy as np

lower_green=np.array([65, 60, 60])
higher_green= np.array([80,255,255])
filenames=[]
for image in os.listdir("/home/ruchika/Documents/Summer of Science ML/traffic_light/1/"):
	filenames.append(image)
print(filenames)
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/1')
for i in range(0, len(filenames)):
	print(i)
	print(filenames[i])
	traffic= cv2.imread(filenames[i])
	hsv= cv2.cvtColor(traffic, cv2.COLOR_BGR2HSV)
	thresh= cv2.inRange(hsv, lower_green, higher_green)
	new_image= thresh
	old_name= filenames[i].split('.')[0]
	new_name= old_name +'onlyGreen.jpg'
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen31')
	cv2.imwrite(new_name, new_image)
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/1')
