import cv2
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


for i in range (1,10):
	fold = str(i)+'/'
	dst_fold = str(i)+'a/a'
	images = []
	for (dirpath, dirnames, filenames) in os.walk(fold):
		for name in filenames:
			if (name.endswith('.jpg')):
				images.append(name)
	print(images)
	for image in images:
		load = cv2.imread(fold + image)
		load = cv2.flip(load,1)
		ensure_dir(dst_fold+image)
		cv2.imwrite(dst_fold+image,load)























