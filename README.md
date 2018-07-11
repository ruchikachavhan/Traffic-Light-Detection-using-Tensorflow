# Traffic-Light-Detection-using-Tensorflow
Detection of the direction of green light arrow(up, right and left).
# Network 
thresholding green out of the images -> resizing images into 100*100 ->  three layered convolutinal network.

Examples of images in training set


![29](https://user-images.githubusercontent.com/32021556/42445981-4c14c88c-8364-11e8-83f7-0138512418ba.png)
![29onlygreen](https://user-images.githubusercontent.com/32021556/42446054-858cdb7c-8364-11e8-884c-86d8b7a01ba6.jpg)
Arrow pointing upwards

![4832](https://user-images.githubusercontent.com/32021556/42446154-cf591784-8364-11e8-90bc-eeb31514f100.jpg)
![4832onlygreen](https://user-images.githubusercontent.com/32021556/42446199-f132af3c-8364-11e8-9afc-c3d0a5de45e8.jpg)
Arrow pointing towards left

Results:

![selection_021](https://user-images.githubusercontent.com/32021556/42467021-43f2521e-83a0-11e8-80e0-026921a0ce0a.png)
                                           By applying threshold -> 
![selection_020](https://user-images.githubusercontent.com/32021556/42466363-3ba7f188-839e-11e8-8e48-970fe6827f9c.png)


Result:
[[0.45063314 0.51786035 0.03150653]]


Probability that the arrow points upwards= 0.45063314


Probabilty that the arrow points towards left= 0.51786035


Probabilty that the arrow points towards right= 0.03150653
