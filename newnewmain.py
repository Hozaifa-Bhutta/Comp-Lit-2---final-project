#oldmain.py
#YOU WERE WORKING ON CHANGING THE MODEL
#This should be the main model where the training occurs
# Import libraries

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
import pandas as pd
from create_batch import CreateBatch
import sys
from numba import jit, cuda 

np.set_printoptions(threshold=sys.maxsize)

#Maximum is 255
#Minimum is 0







#print (train_y.shape, test_y.shape)

training_iter = 1000
#Start off at 0.001, then 0.003, then 0.01 etc...
learning_rate = 0.001 
#should be a power of 2
batch_size = 64

#Number of classes - 41 in real dataset
n_classes = 40
n_channels = 3
#two placeholders, x and y
#First value is left as 'None' as it'll be defined later on as 'batch_size'
x = tf.placeholder('float', [None, 385, 165, n_channels])
y = tf.placeholder('float', [None, n_classes])


#create labels
#for i in range(46):
Phonemes = {}
full_script = []
for i in range(1,27):
	with open('CSV_files/min_' + str(i) + '.csv', 'r') as f:
		reader = csv.reader(f)
		list_min = list(reader)
		full_script += list_min


number = 0
for i in full_script:
	if number ==40:
		break

	if i[0][-2] == '_':
		if (i)[0][:-2] not in Phonemes:
			Phonemes[(i)[0][:-2]] = number
			number += 1
	if i[0] == 'silence':
		if (i)[0] not in Phonemes:
			Phonemes[(i)[0]] = number
			number += 1

	else:
		pass


print (len(full_script))
print (Phonemes)

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
	'wc1': numpy.load("weights_wc1")
	'wc2': numpy.load("weights_wc2"),
	'wc3': numpy.load("weights_wc3"),

	#For fully conncected
	#Shape first parameter equals result of previous output
	'wd1': numpy.load("weights_wd1"),
	# For output
	'out': numpy.load("weights_out")
}

biases = {
	#All the biases for the NN model
	#Just like the weights, these values must be intialized

	'bc1':numpy.load("biases_bc1"),
	'bc2':numpy.load("biases_bc2"),
	'bc3':numpy.load("biases_bc3"),

	'bd1':numpy.load("biases_bd1"),
	'out':numpy.load("biases_out")
}


def conv_net(x, weights, biases):
	#This is the whole model
	
	# Performs convolution, does bias, and relu. Calls the conv2d function defined above
	conv1 = conv2d(x,weights['wc1'],biases['bc1'])

	# Performs max_pooling. Calls maxpooling function defined above
	conv1 = maxpool2d(conv1)
	print (conv1.shape)
	#Covloution layer 2 
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2)
	print (conv2.shape)

	#Convloution layer 3
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3)
	print (conv3.shape)

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


#Optimazion method

#Evaluation
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))


#Initialize all variables
init = tf.global_variables_initializer()

sess = tf.Session()
#Start computational graph
sess.run(init)

#for frame in range(3600,6000):
 #   z = random.randint(2000,156000) #should be ay
 #   while z%6000<100 or z%6000>5900 or full_script[z][0] =='not-found-in-audio' or full_script[z][0] =='oov':

 #       z = random.randint(0,156000)

    
    
    
#Specify frame here
z = 224
img = framenumToimg(z)/255
batch_x = img
batch_x = np.reshape(batch_x, (1,385,413,3))
if full_script[z-1][0][-2] == '_':
    _y = (Phonemes[full_script[z-1][0][:-2]])
else:
    _y = (Phonemes[full_script[z-1][0]])



prediction = sess.run(pred, feed_dict = {x:batch_x})
correct_prediction = tf.argmax(prediction, 1)
print ('\n')
print (z)
print (sess.run(correct_prediction))
print (_y)

print (sess.run(correct_prediction) == _y)
