import tensorflow as tf
import cv2
import numpy as np
import os

#Number of classes=3(right, left,up)
#Defining training examples and labels
right_arrow=[]
left_arrow=[]
up_arrow=[]
error= 'error.txt'
error_file= open(error, 'w')
#Reading images
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1'):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (100,100))
	up_arrow.append(re)
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2'):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (100,100))
	left_arrow.append(re)
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3')
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3'):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (100,100))
	right_arrow.append(re)

total_examples= len(right_arrow)+ len(left_arrow)+ len(up_arrow)
labels= np.zeros([total_examples, 3])
examples=[]
# Labels
for i in range(0, len(up_arrow)):
	examples.append(up_arrow[i])
	labels[i][0]=1
for j in range(len(up_arrow), len(up_arrow)+len(left_arrow)):
	examples.append(left_arrow[j-len(up_arrow)])
	labels[j][1]=1
for k in range(len(up_arrow)+len(left_arrow), len(up_arrow)+len(left_arrow)+len(right_arrow)):
	examples.append(right_arrow[k-len(left_arrow)-len(up_arrow)])
	labels[k][2]=1

#Defining hyperparameters of neural network 
learning_rate= 0.0001
epochs= 30
print(labels[0].shape)
#Defining model of the neural network
def create_new_conv_layer(input, channels, filters, filter_shape, pool_shape, name):
	convolutional_shape= [filter_shape[0], filter_shape[1], channels, filters]
	weights= tf.get_variable(initializer= tf.truncated_normal(convolutional_shape, stddev= 0.03), name= name+'w')
	bias= tf.get_variable(initializer=tf.truncated_normal([filters]), name= name+'b')
	out_layer= tf.nn.conv2d(input, weights, [1,1,1,1], padding='SAME')
	out_layer=out_layer+bias
	out_layer= tf.nn.relu(out_layer)
	ksize=[1, pool_shape[0], pool_shape[1], 1]
	strides=[1,2,2,1]
	out_layer= tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
	return out_layer

#Defining placeholders
x = tf.placeholder(tf.float32, [100,100])
input_x = tf.reshape(x, [-1, 100,100, 1])
y = tf.placeholder(tf.float32, [3])

#Calculating convolutional layer
layer1 = create_new_conv_layer(input_x, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
layer3 = create_new_conv_layer(layer2, 64, 128, [5, 5], [2, 2], name='layer3')
print(layer3)
#Flattening 
flattened = tf.reshape(layer3, [-1,13*13*128 ])

#Defining the rest of the model
#Layer1
weights1 = tf.get_variable(initializer=tf.truncated_normal([13*13*128, 1000], stddev=0.03), name='weights1')
b1= tf.get_variable(initializer= tf.truncated_normal([1000], stddev=0.01), name='b1')
dense_layer1 =tf.add( tf.matmul(flattened, weights1), b1)
dense_layer1 = tf.nn.relu(dense_layer1)
#Layer2
weights2 = tf.get_variable(initializer=tf.truncated_normal([1000, 3], stddev=0.03), name='weights2')
b2 = tf.get_variable(initializer= tf.truncated_normal([3], stddev=0.01), name='b2')
dense_layer2 = tf.add(tf.matmul(dense_layer1, weights2), b2)
y_ = tf.nn.softmax(dense_layer2)
#Calculating cross- entropy
print("dense_layer2",tf.transpose(dense_layer2))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
sess= tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    avg_cost = 0.0
    for i in range(0, len(examples)):
    	t=[]
    	for ty in range (0, len(labels[0])):
    		t.append(labels[i][ty])
    	_, c= sess.run([optimiser, cross_entropy], feed_dict= {x: examples[i], y:t})
    	avg_cost=avg_cost+c
    print(avg_cost)
    error_file.write(str(avg_cost))
    error_file.write("\n")
