#newmain.py
#YOU WERE WORKING ON CHANGING THE MODEL
#This should be the main model where the training occurs
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#The input_data is not required in the final phase since we will be making our own data set
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



training_iter = 1000
#Start off at 0.001, then 0.003, then 0.01 etc...
learning_rate = 0.001
#should be a power of 2
batch_size = 64

#Number of classes - 41 
n_classes = 41
n_channels = 3
#two placeholders, x and y
#First value is left as 'None' as it'll be defined later on as 'batch_size'
x = tf.placeholder('float', [None, 385, 413, n_channels])
y = tf.placeholder('float', [None, n_classes])


#create labels
Phonemes = {}
full_script = []

for i in range(1,27):
	with open('labels/min_' + str(i) + '.csv', 'r') as f:
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




#Creates base_batch_x and base_batch_y
base_fake_batch_x = []
base_fake_batch_y = []
for training_ex in range(256):

	z = random.randint(1,156000)
	while z%6000<100 or z%6000>5900 or full_script[z][0] =='not-found-in-audio' or full_script[z][0] =='oov':
		z = random.randint(1,156000)
	img = framenumToimg(z)/255
	#print (img)


	base_fake_batch_x.append(img)
	base_fake_batch_x[training_ex] = np.reshape(base_fake_batch_x[training_ex], (1,385,413,3))
	if full_script[z-1][0][-2] == '_':
		base_fake_batch_y.append(Phonemes[full_script[z-1][0][:-2]])
	else:
		base_fake_batch_y.append(Phonemes[full_script[z-1][0]])

print ('created base_fake_batch_x with a length of ' + str(len(base_fake_batch_x))+ ' and created base_fake_batch_y with a length of ' + str(len(base_fake_batch_y)))

for fake_i in range(256):
	if fake_i == 0:
		base_batch_x = np.array(base_fake_batch_x[0])
	else:
		base_batch_x = np.concatenate((base_batch_x, base_fake_batch_x[fake_i]),0)
print ('created base_batch x with a shape of ' + str(base_batch_x.shape))
assistant_y = np.zeros((256,41))
base_fake_batch_y = np.array(base_fake_batch_y)

base_batch_y = assistant_y[np.arange(256),base_fake_batch_y] = 1
base_batch_y = assistant_y




thirty_second_window = 3000


for i in range(training_iter):
	for batch in range(1280//batch_size):
		fake_batch_x = []
		fake_batch_y = []

		
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




		#batch_x is defined as a numpy array of fake_batch_x and batch_y is defined    
		for fake_i in range(batch_size):
			if fake_i == 0:
				batch_x = np.array(fake_batch_x[0])
			else:
				batch_x = np.concatenate((batch_x, fake_batch_x[fake_i]),0)
		print ('created batch x with a shape of ' + str(batch_x.shape))
		assistant_y = np.zeros((batch_size,41))
		fake_batch_y = np.array(fake_batch_y)
		batch_y = assistant_y[np.arange(batch_size),fake_batch_y] = 1
		batch_y = assistant_y



		# Optimization is run
		print ('running batch x for batch num: ' +str(batch))
		prev_weights = weights['wc1']
		opt = sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
		print (weights['wc1'] == prev_weights)
		#prediction = sess.run(pred, feed_dict = {x:batch_x})
		if batch %9 ==0:
			#Saves weights every 9 batches
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
	#SOMETHING WRONG HERE
	#SOMETHING WRONG HERE
	#SOMETHING WRONG HERE
	#SOMETHING WRONG HERE
	#SOMETHING WRONG HERE


	loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})
	base_loss, base_acc = sess.run([cost, accuracy], feed_dict={x:base_batch_x, y:base_batch_y})

	#Loss and accuracy 
	print ('Iter ' + str(i))
	print ('Optimazion finished') 
	print ('Training Loss: ' + str(loss))
	print ('Training Accuracy: ' + str(acc))
	print ('Base Accuracy: ' + str(base_acc))
	print ('Base Loss: ' + str(base_loss))
	print ('window of time ' + str(thirty_second_window))
	if acc >= 0.9:
		thirty_second_window += 3000


	#print ('Test Loss: ' + str(test_loss))
	#print ('Test Accuracy: ' +str(valid_acc))
