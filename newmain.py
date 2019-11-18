#newmain.py
#YOU WERE WORKING ON CHANGING THE MODEL
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
from frametoimg import framenumToimg
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

IMG_DIR = 'full_vid'
#full_list = []
'''for img in os.listdir(IMG_DIR):
		img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

		img_array = (img_array.flatten())

		img_array  = img_array.reshape(-1, 1).T
		img_array = list(img_array[0])
		#for i in range(0,len(img_array)):
		#	img_array[i] = float(img_array[i])*(1/255)
		full_list.append((img_array))
print (full_list[0])'''
def read_img(img):
	im = cv2.imread("/Users/Hozai/Desktop/audios/raw_pictures/"+img,1)
	print (type(im)) #Print <class 'numpy.ndarray'>
	print (im.size) #prints 2100000
	return im
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

training_iter = 1000
#Start off at 0.001, then 0.003, then 0.01 etc...
learning_rate = 0.001 
#should be a power of 2
batch_size = 64

#Number of classes - 41 in real dataset
n_classes = 41
n_channels = 3
#two placeholders, x and y
#First value is left as 'None' as it'll be defined later on as 'batch_size'
x = tf.placeholder('float', [None, 385, 413, n_channels])
y = tf.placeholder('float', [None, n_classes])


#create labels
#for i in range(46):
Phonemes = {}
full_script = []
with open('labels/min_1.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_2.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_3.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_4.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_5.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_6.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_7.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_8.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_9.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_10.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_11.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_12.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_13.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_14.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_15.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_16.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_17.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_18.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_19.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_20.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_21.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_22.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_23.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_24.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_25.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min
with open('labels/min_26.csv', 'r') as f:
	reader = csv.reader(f)
	list_min = list(reader)
	full_script += list_min

number = 0
for i in full_script:
	if number >=1000:
		break

	if i[0][-2] == '_':
		if (i)[0][:-2] not in Phonemes:
			Phonemes[(i)[0][:-2]] = number
			number += 1
	else:
		if (i)[0] not in Phonemes:
			Phonemes[(i)[0]] = number
			number +=1



		#print (Phonemes)


#
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
	'wc1': tf.get_variable('W0', shape = (3,3,3,27), initializer= tf.contrib.layers.xavier_initializer()),
	'wc2': tf.get_variable('W1', shape = (3,3,27,54), initializer= tf.contrib.layers.xavier_initializer()),
	'wc3': tf.get_variable('W2', shape = (3,3,54,108), initializer= tf.contrib.layers.xavier_initializer()),
	'wc4': tf.get_variable('W3', shape = (3,3,108,216), initializer= tf.contrib.layers.xavier_initializer()),
	'wc5': tf.get_variable('W4', shape = (3,3,216,216), initializer= tf.contrib.layers.xavier_initializer()),

	#For fully conncected
	#Shape first parameter equals result of previous output
	#4 by 4 image with 128 channels
	'wd1': tf.get_variable('W5', shape = (13*13*216, 216), initializer= tf.contrib.layers.xavier_initializer()),
	# For output
	'out': tf.get_variable('W6', shape = (216, n_classes), initializer= tf.contrib.layers.xavier_initializer())
}

biases = {
	#All the biases for the NN model
	#Just like the weights, these values must be intialized

	'bc1':tf.get_variable('B0', shape = 27, initializer=tf.contrib.layers.xavier_initializer()),
	'bc2':tf.get_variable('B1', shape = 54, initializer=tf.contrib.layers.xavier_initializer()),
	'bc3':tf.get_variable('B2', shape = 108, initializer=tf.contrib.layers.xavier_initializer()),
	'bc4':tf.get_variable('B3', shape = 216, initializer=tf.contrib.layers.xavier_initializer()),
	'bc5':tf.get_variable('B4', shape = 216, initializer=tf.contrib.layers.xavier_initializer()),

	'bd1':tf.get_variable('B5', shape = 216, initializer=tf.contrib.layers.xavier_initializer()),
	'out':tf.get_variable('B6', shape = n_classes, initializer=tf.contrib.layers.xavier_initializer())
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
	#Convloution layer 4
	conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
	conv4 = maxpool2d(conv4)
	#Convloution layer 5
	conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
	conv5 = maxpool2d(conv5)
	#Fully connected layer
	# Reshapes last layer accordingly

	conv5_flattened = tf.reshape(conv5,[-1, weights['wd1'].get_shape()[0]])
	#Multiplies fc1 and 'wd1' and then adds it with bias
	fc1 = tf.add(tf.matmul(conv5_flattened, weights['wd1']),biases['bd1'])
	#Applies relu
	fc1 = tf.nn.relu(fc1)

	#Output layer
	out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])
	out = tf.nn.softmax(tf.add(tf.matmul(fc1,weights['out']), biases['out']))
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
thirty_second_window = 3000
for i in range(training_iter):
	for batch in range(1280//batch_size):
		fake_batch_x = []
		fake_batch_y = []
		#batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
		
		'''a = np.array([0,1])
		b = np.zeros((2, 2))
		b[np.arange(2), a] = 1'''
		
		#Runs backpropagation
		#Feeds placeholder x and y

		#puts images into fake_batch_x list
		for training_ex in range(batch_size):
			z = random.randint(thirty_second_window - 3000,thirty_second_window)
			while z%6000<100 or z%6000>5900 or full_script[z][0] =='not-found-in-audio' or full_script[z][0] =='oov':
				z = random.randint(thirty_second_window-3000,thirty_second_window)
			img = framenumToimg(z)/255
			#print (img)


			fake_batch_x.append(img)
			fake_batch_x[training_ex] = np.reshape(fake_batch_x[training_ex], (1,385,413,3))
			if full_script[z-1][0][-2] == '_':
				fake_batch_y.append(Phonemes[full_script[z-1][0][:-2]])
			else:
				fake_batch_y.append(Phonemes[full_script[z-1][0]])
			#print ('using frame ' + str(z) + ' and the label is ' + str(fake_batch_y[-1]))

		print ('created fake_batch_x with a length of ' + str(len(fake_batch_x))+ ' and created fake_batch_y with a length of ' + str(len(fake_batch_y)) + ' for batch ' + str(batch))

		#batch_x is defined as a numpy array of fake_batch_x		
		for fake_i in range(batch_size):
			if fake_i == 0:
				batch_x = np.array(fake_batch_x[0])
			else:
				batch_x = np.concatenate((batch_x, fake_batch_x[fake_i]),0)
		print ('created batch x with a shape of ' + str(batch_x.shape))
		assistant_y = np.zeros((batch_size,41))
		fake_batch_y = np.array(fake_batch_y)
		#print (np.arange(batch_size))
		#print (fake_batch_y)
		batch_y = assistant_y[np.arange(batch_size),fake_batch_y] = 1
		batch_y = assistant_y
		#print (batch_y)
		print ('running batch x for batch num: ' +str(batch))
		opt = sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
		prediction = sess.run(pred, feed_dict = {x:batch_x})
		#print (prediction)
		#print (batch_y)
		#print (z)
		#print (fake_batch_y)
		#Runs Evaluation
		if batch %9 ==0:
			print (batch)
			print ('saving weights')
			np.save('weights_biases/weight_1.npy', sess.run(weights['wc1']))
			np.save('weights_biases/weight_2.npy', sess.run(weights['wc2']))
			np.save('weights_biases/weight_3.npy', sess.run(weights['wc3']))
			np.save('weights_biases/weight_4.npy', sess.run(weights['wc4']))
			np.save('weights_biases/weight_5.npy', sess.run(weights['wc5']))
			np.save('weights_biases/weight_wd1.npy', sess.run(weights['wd1']))
			np.save('weights_biases/weight_out.npy', sess.run(weights['out']))
			np.save('weights_biases/bias_1.npy', sess.run(biases['bc1']))
			np.save('weights_biases/bias_2.npy', sess.run(biases['bc2']))
			np.save('weights_biases/bias_3.npy', sess.run(biases['bc3']))
			np.save('weights_biases/bias_4.npy', sess.run(biases['bc4']))
			np.save('weights_biases/bias_5.npy', sess.run(biases['bc5']))
			np.save('weights_biases/bias_wd1.npy', sess.run(biases['bd1']))
			np.save('weights_biases/bias_out.npy', sess.run(biases['out']))
		print ('finished batch number ' + str(batch))
		print ('\n')
		previous_batch_x = batch_x
		previous_batch_y = batch_y
	#Loss and accuracy for train set
	#FIX THIS!
	#FIX THIS!
	#FIX THIS!
	#More than just batch
	test_batch_x = batch_x = np.concatenate((batch_x, previous_batch_x),0)
	test_batch_y = batch_x = np.concatenate((batch_y, previous_batch_y),0)	
	loss, acc = sess.run([cost, accuracy], feed_dict={x:test_batch_x, y:test_batch_y})
	#Loss and accuracy for test set
	#test_loss, valid_acc = sess.run([cost,accuracy], feed_dict={x:test_X,y:test_y})
	print ('Iter ' + str(i))
	print ('Optimazion finished') 
	print ('Training Loss: ' + str(loss))
	print ('Training Accuracy: ' + str(acc))
	if acc >= 0.9:
		thirty_second_window += 3000
	#print ('Test Loss: ' + str(test_loss))
	#print ('Test Accuracy: ' +str(valid_acc))
