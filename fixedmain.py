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
	'wc1': tf.get_variable('W0', shape = (3,3,3,60), initializer= tf.contrib.layers.xavier_initializer()),
	'wc2': tf.get_variable('W1', shape = (3,3,60,120), initializer= tf.contrib.layers.xavier_initializer()),
	'wc3': tf.get_variable('W2', shape = (3,3,120,240), initializer= tf.contrib.layers.xavier_initializer()),

	#For fully conncected
	#Shape first parameter equals result of previous output
	'wd1': tf.get_variable('W3', shape = (49*21*240, 240), initializer= tf.contrib.layers.xavier_initializer()),
	# For output
	'out': tf.get_variable('W4', shape = (240, n_classes), initializer= tf.contrib.layers.xavier_initializer())
}

biases = {
	#All the biases for the NN model
	#Just like the weights, these values must be intialized

	'bc1':tf.get_variable('B0', shape = 60, initializer=tf.zeros_initializer()),
	'bc2':tf.get_variable('B1', shape = 120, initializer=tf.zeros_initializer()),
	'bc3':tf.get_variable('B2', shape = 240, initializer=tf.zeros_initializer()),

	'bd1':tf.get_variable('B3', shape = 240, initializer=tf.zeros_initializer()),
	'out':tf.get_variable('B4', shape = n_classes, initializer=tf.zeros_initializer())
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


multiplier = 1




#Test batch
fake_batch_x = []
fake_batch_y = []
'''
for training_ex in range(248):
	z = random.randint(1,156000)

	while not z > (151000) or not z< (155000) or full_script[z-1][0] == 'oov' or full_script[z-1][0] == 'not-found-in-audio' or full_script[z-1][0] == 'silence' or z%6000 < 50 or z%6000 >5970 :
		z = random.randint(0,156000)

	#Else it will pick one from before the new batch, therefore it can't forget old stuff

	img = np.load('all_spectograms/img_'+str(z)+'.npy')/255
	if (img.shape != (385,165,3)):
		continue 

	fake_batch_x.append(img)
	print (len(fake_batch_x))
	exit()
	fake_batch_x[training_ex] = np.reshape(fake_batch_x[training_ex], (1,385,165,3))
	if full_script[z-1][0][-2] == '_':
		fake_batch_y.append(Phonemes[full_script[z-1][0][:-2]])
	else:
		fake_batch_y.append(Phonemes[full_script[z-1][0]])
	#print ('using frame ' + str(z) + ' and the label is ' + str(fake_batch_y[-1]))

print ('created fake_batch_x with a length of ' + str(len(fake_batch_x))+ ' and created fake_batch_y with a length of ' + str(len(fake_batch_y)))

#batch_x is defined as a numpy array of fake_batch_x		
for fake_i in range(248):
	if fake_i == 0:
		batch_x = np.array(fake_batch_x[0])
	else:
		batch_x = np.concatenate((batch_x, fake_batch_x[fake_i]),0)
print ('created batch x with a shape of ' + str(batch_x.shape))
assistant_y = np.zeros((248,n_classes))
fake_batch_y = np.array(fake_batch_y)


batch_y = assistant_y[np.arange(248),fake_batch_y] = 1
batch_y = assistant_y

test_batch_y = batch_y
test_batch_x = batch_x'''





#with tf.device('/gpu:0'):
for i in range(training_iter):
	for batch in range(0,1280//batch_size):
		batch_x , batch_y = CreateBatch(batch_size,multiplier, full_script, Phonemes, n_classes, batch)



		print ('running batch x for batch num: ' +str(batch))
		opt = sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
		prediction = sess.run(pred, feed_dict = {x:batch_x})
		#Runs Evaluation
		if (batch+1) %10 ==0:
			print ('saving weights')
			np.save('weight_1.npy', sess.run(weights['wc1']))
			np.save('weight_2.npy', sess.run(weights['wc2']))
			np.save('weight_3.npy', sess.run(weights['wc3']))
			np.save('weight_wd1.npy', sess.run(weights['wd1']))
			np.save('weight_out.npy', sess.run(weights['out']))
			np.save('bias_1.npy', sess.run(biases['bc1']))
			np.save('bias_2.npy', sess.run(biases['bc2']))
			np.save('bias_3.npy', sess.run(biases['bc3']))
			np.save('bias_wd1.npy', sess.run(biases['bd1']))
			np.save('bias_out.npy', sess.run(biases['out']))
		print ('finished batch number ' + str(batch))
		print ('\n')
	batch = 'training batch'
	#Loss and accuracy for train set
	#FIX THIS!
	#FIX THIS!
	#FIX THIS!
	#More than just batch
	#train_batch_x , train_batch_y = CreateBatch(batch_size,multiplier, full_script, Phonemes, n_classes, batch)
	loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})
	
	#test_loss,test_acc = sess.run([cost, accuracy], feed_dict={x:test_batch_x, y:test_batch_y})
	#test_losses.append(test_loss)
	#test_accs.append(test_acc)
	#Loss and accuracy for test set
	#test_loss, valid_acc = sess.run([cost,accuracy], feed_dict={x:test_X,y:test_y})
	print ('\n\n\n')
	print ('Iter ' + str(i))
	print ('Optimazion finished') 
	print ('Training Loss: ' + str(loss))
	print ('Training Accuracy: ' + str(acc))
	print ('Multiplier: ' + str(multiplier))
	#print ('Testing Loss: ' + str(test_loss))
	#print ('Testing Accuracy: ' + str(test_acc))
	#if i%10 ==0:
	#	print (test_losses)
	#	print (test_accs)
	print ('\n\n\n')
	if acc >0.92:
		try:
			acc_check += 1
			if acc_check > 3:
				multiplier +=1
				acc_check = 1

		except:
			acc_check = 1
	#print ('Test Loss: ' + str(test_loss))
	#print ('Test Accuracy: ' +str(valid_acc))
