#newmain.py
#This should be the main model where the training occurs
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#The input_data is not required in the final phase since we will be making our own data set
from tensorflow.examples.tutorials.mnist import input_data
# Only works for IPython --->      %matplotlib inline
#To run 'magic'
import os
import random
from reads_csv import read_file
import csv

IMG_DIR = 'full_vid'
full_list = []
for img in os.listdir(IMG_DIR):
		img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

		img_array = (img_array.flatten())

		img_array  = img_array.reshape(-1, 1).T
		img_array = list(img_array[0])
		for i in range(0,len(img_array)):
			img_array[i] = float(img_array[i])*(1/255)
		full_list.append((img_array))
print (full_list[0])
#Maximum is 255
#Minimum is 0
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
#Imports data from tensorflow library 'input_data'. one_hot transforms categorical labels into binary vectors
'''one_hot - In one-hot encoding, you convert the categorical data into a vector of numbers. You do this because machine learning algorithms can't work with categorical data directly. 
Instead, you generate one boolean column for each category or class. Only one of these columns could take on the value 1 for each sample. That explains the term "one-hot encoding".'''
#data = input_data.read_data_sets('data/fashion', one_hot = True)


#All images are scaled between 0-1 in this dataset. In artifical dataset we need to scale it down to this value

# Shapes of training set
#print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
#print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

#New line
#print ('\n')

# Shapes of test set
#print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
#print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))

# Create dictionary of target classes
# This will be changed later on into 41 different Phonemes plus 'silence'
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}



# Gives vector version of image '0'
#  print(data.train.images[0]) <--- returns a long vector

# Prints largest value, should be 1
#print(np.max(data.train.images[0]))

#Prints smallest value, should be 0
#print(np.min(data.train.images[0]))

#Reshapes each image into a vector
# The '-1' means that it infers the batch size
#train_X = data.train.images.reshape(-1,28,28,1)
#test_X = data.test.images.reshape(-1,28,28,1)


'''
image_0 = train_X[0]
image_1=train_X[1]
image_2=train_X[2]
z=tf.concat([image_0,image_1,image_2],2)
print (z.shape)'''


# Should return 'batch size' (inferred) by 28 by 28 by 1
#print (train_X.shape, test_X.shape)

# Extracts 'test' dataset
#train_y = data.train.labels
#test_y = data.test.labels

#print (train_y.shape, test_y.shape)

training_iter = 200
#Start off at 0.001, then 0.003, then 0.01 etc...
learning_rate = 0.001 
#should be a power of 2
batch_size = 64

#Number of classes - 41 in real dataset
n_classes = 2

#two placeholders, x and y
#First value is left as 'None' as it'll be defined later on as 'batch_size'
x = tf.placeholder('float', [None, 480, 480, None])
y = tf.placeholder('float', [None, n_classes])


def conv2d(x, W, b, stride = 1):
	#A conv2d rapper that performs convolution, adds bias, and does relu
	# x - input, W - weights, b - bias

	#Performs convolution
	# First and last values for 'strides' must be 1 as the first one represents filter 
	# going through each  image and the last one represents each channel
	conv1 = tf.nn.conv2d(x, W, strides= [1,stride,stride,1], padding = "SAME")

	# Adds bias
	conv1_b = tf.nn.bias_add(conv1, b)

	#Performs relu and returns
	return tf.nn.relu(conv1_b)

def maxpool2d(x, k=2):
	# Performs maxpooling
	return tf.nn.max_pool(x, ksize=(1,k,k,1),strides = (1,k,k,1),padding = 'SAME')

weights = {
	#All the weights for the NN model
	#The values are variables so they must be intialized

	#'shape' parameters - filter size, input dimension, output dimension

	#Convolution
	'wc1': tf.get_variable('W0', shape = (3,3,30,60), initializer= tf.contrib.layers.xavier_initializer()),
	'wc2': tf.get_variable('W1', shape = (3,3,32,120), initializer= tf.contrib.layers.xavier_initializer()),
	'wc3': tf.get_variable('W2', shape = (3,3,64,240), initializer= tf.contrib.layers.xavier_initializer()),
	#For fully conncected
	#Shape first parameter equals result of previous output
	#4 by 4 image with 128 channels
	'wd1': tf.get_variable('W3', shape = (60*60*240, 128), initializer= tf.contrib.layers.xavier_initializer()),
	# For output
	'out': tf.get_variable('W4', shape = (128, n_classes), initializer= tf.contrib.layers.xavier_initializer())
}

biases = {
	#All the biases for the NN model
	#Just like the weights, these values must be intialized

	'bc1':tf.get_variable('B0', shape = 60, initializer=tf.contrib.layers.xavier_initializer()),
	'bc2':tf.get_variable('B1', shape = 120, initializer=tf.contrib.layers.xavier_initializer()),
	'bc3':tf.get_variable('B2', shape = 240, initializer=tf.contrib.layers.xavier_initializer()),
	'bd1':tf.get_variable('B3', shape = 128, initializer=tf.contrib.layers.xavier_initializer()),
	'out':tf.get_variable('B4', shape = n_classes, initializer=tf.contrib.layers.xavier_initializer())
}


def conv_net(x, weights, biases):
	#This is the whole model
	
	# Performs convolution, does bias, and relu. Calls the conv2d function defined above
	conv1 = conv2d(x,weights['wc1'],biases['bc1'])

	# Performs max_pooling. Calls maxpooling function defined above
	conv1 = maxpool2d(conv1)

	#Covloution layer 2 
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2)

	#Convloution layer 3
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3)

	#Fully connected layer
	# Reshapes last layer accordingly
	conv3_flattened = tf.reshape(conv3,[-1, weights['wd1'].get_shape()[0]])
	#Multiplies fc1 and 'wd1' and then adds it with bias
	fc1 = tf.add(tf.matmul(conv3_flattened, weights['wd1']),biases['bd1'])
	#Applies relu
	fc1 = tf.nn.relu(fc1)

	#Output layer
	out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])
	print (out.shape)
	return out

# Predictions are  stored here
pred = conv_net(x, weights, biases)

# Cost is the 'average' of the loss

loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(loss)

#Optimazion method
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluation
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#Initialize all variables
init = tf.global_variables_initializer()

sess = tf.Session()
#Start computational graph
sess.run(init)
train_loss = []
test_loss = []
train_accuracy = []
test_loss = []
summary_writer = tf.summary.FileWriter('./Output',sess.graph)
for i in range(training_iter):
	for batch in range(1194//batch_size):
		fake_batch_x = []
		fake_batch_y = []
		#batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
		
		'''a = np.array([0,1])
		b = np.zeros((2, 2))
		b[np.arange(2), a] = 1'''
		
		#Runs backpropagation
		#Feeds placeholder x and y
		for training_ex in range(batch_size):
			z = random.randint(20,1170)
			frames = []
			for frame_index in range((z-14),(z+14)):
				frame = np.array(full_list[frame_index])
				file = np.reshape(file,[-1,480,480,1])
				frames.append[file]
			real_training_ex = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], frames[6], frames[7], frames[8], frames[9], frames[10], frames[11], frames[12],frames[13], frames[14], frames[15], frames[16], frames[17], frames[18], frames[19], frames[20], frames[21], frames[22], frames[23], frames[24], frames[25], frames[26], frames[27], frames[28], frames[29]),3)       

			fake_batch_x.append(real_training_ex)
			if z <= 599:
				fake_batch_y.append(0)
			else:
				fake_batch_y.append(1)
		for fake_i in range(batch_size):
			if fake_i == 0:
				batch_x = fake_batch_x[0]
			else:
				batch_x = np.concatenate((batch_x, fake_batch_x[fake_i]))
		assistant_y = np.zeros((batch_size,2))
		fake_batch_y = np.array(fake_batch_y)
		batch_y = assistant_y[np.arange(batch_size),fake_batch_y] = 1
		batch_y = assistant_y
		opt = sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
		prediction = sess.run(pred, feed_dict = {x:batch_x})
		#Runs Evaluation
		if batch %100 ==0:
			print (batch)
	#Loss and accuracy for train set
	#FIX THIS!
	#FIX THIS!
	#FIX THIS!
	#More than just batch
	loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})
	#Loss and accuracy for test set
	#test_loss, valid_acc = sess.run([cost,accuracy], feed_dict={x:test_X,y:test_y})
	print ('Iter ' + str(i))
	print ('Optimazion finished') 
	print ('Training Loss: ' + str(loss))
	print ('Training Accuracy: ' + str(acc))
	#print ('Test Loss: ' + str(test_loss))
	#print ('Test Accuracy: ' +str(valid_acc))
