import tensorflow as tf
import cv2
import numpy as np
import os

#Number of classes=3(right, left,up)
#Defining training examples and labels
right_arrow=[]
left_arrow=[]
up_arrow=[]

#Reading images
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1_resized') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen1'):
	img= cv2.imread(image)
	up.append()
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2_resized') 
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen2'):
	img= cv2.imread(image)
	left_arrow.append()
os.chdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3_resized')
for image in os.listdir('/home/ruchika/Documents/Summer of Science ML/traffic_light/onlyGreen3'):
	img= cv2.imread(image)
	right_arrow.append()

total_examples= len(right_arrow)+ len(left_arrow)+ len(up_arrow)
labels= np.zeros([total_examples, 3])
examples=[]
# Labels
for i in range(0, len(up_arrow)):
	examples.append(up_arrow[i])
	labels[i][0]=1
for i in range(0, len(left_arrow)):
	examples.append(left_arrow[i])
	labels[i][1]=1
for i in range(0, len(right_arrow)):
	examples.append(right_arrow[i])
	labels[i][2]=1

#Defining hyperparameters of neural network 
learning_rate= 0.001
epochs= 10

#Defining model of the neural network
def create_convolutinal layers(input, channels, filters, filter_shape, pool_shape, name):
	convolutional_shape =[filter_shape[0], filter_shape[1], channels, filters]
	weights = tf.Variable(tf.truncated_normal(convolutional_shape, stddev=0.03),name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([filters]), name=name+'_b')
	# setup the convolutional layer 
    out_layer = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME')
	# add the bias
    out_layer += bias
	# apply ReLu
    out_layer = tf.nn.relu(out_layer)
	# now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
	return out_layer  
#Defining placeholders
x = tf.placeholder(tf.float32, [None, 784])
# input_x = tf.reshape(x, [-1, 28,28, 1])
y = tf.placeholder(tf.float32, [None, 3])

#Calculating convolutional layer
layer1 = create_new_conv_layer(x, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

#Flattening 
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

#Defining the rest of the model
#Layer1
weights1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
b1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, weights1) + b1
dense_layer1 = tf.nn.relu(dense_layer1)
#Layer2
weights2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, weights2) + b2
y_ = tf.nn.softmax(dense_layer2)
#Calculating cross- entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(0 , len(examples)):
            #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: examples[i], y: labels[i]})
            avg_cost += c 
        print('cost', avg_cost)