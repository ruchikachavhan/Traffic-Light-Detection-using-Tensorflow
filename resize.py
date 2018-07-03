#Resizing all images in training data
import cv2
import os

os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1'):
	img= cv2.imread(image)
	re=cv2.resize(img, (28,28))
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1_resized')
	cv2.imwrite(image, re)
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1')


os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2_resized') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2'):
	img= cv2.imread(image)
	re=cv2.resize(img, (28,28))
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2_resized')
	cv2.imwrite(image, re)
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2')

	
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3_resized')
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3'):
	img= cv2.imread(image)
	re=cv2.resize(img, (28,28))
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3_resized')
	cv2.imwrite(image, re)
	os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3')