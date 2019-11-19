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
print (full_script[224][0])       
print (len(full_script))
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
'''
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
}'''
bias_list = []
bias_list.append(np.load('weights_biases/bias_1.npy'))
bias_list.append(np.load('weights_biases/bias_2.npy'))
bias_list.append(np.load('weights_biases/bias_3.npy'))
bias_list.append(np.load('weights_biases/bias_4.npy'))
bias_list.append(np.load('weights_biases/bias_5.npy'))
bias_list.append(np.load('weights_biases/bias_wd1.npy'))
bias_list.append(np.load('weights_biases/bias_out.npy'))

weight_list = []
weight_list.append(np.load('weights_biases/weight_1.npy'))
weight_list.append(np.load('weights_biases/weight_2.npy'))
weight_list.append(np.load('weights_biases/weight_3.npy'))
weight_list.append(np.load('weights_biases/weight_4.npy'))
weight_list.append(np.load('weights_biases/weight_5.npy'))
weight_list.append(np.load('weights_biases/weight_wd1.npy'))
weight_list.append(np.load('weights_biases/weight_out.npy'))




#for i in range(1 , 6):
   # bc1 = np.load('bias_' + str(i) + '.npy')
'''biases = {
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
'''

def conv_net(x, weights, biases):
    #This is the whole model
    
    # Performs convolution, does bias, and relu. Calls the conv2d function defined above
    conv1 = conv2d(x,weight_list[0],bias_list[0])

    # Performs max_pooling. Calls maxpooling function defined above
    conv1 = maxpool2d(conv1)
    #print (conv1.shape)
    #Covloution layer 2 
    conv2 = conv2d(conv1, weight_list[1], bias_list[1])
    conv2 = maxpool2d(conv2)

    #Convloution layer 3
    conv3 = conv2d(conv2, weight_list[2], bias_list[2])
    conv3 = maxpool2d(conv3)
    #Convloution layer 4
    conv4 = conv2d(conv3, weight_list[3], bias_list[3])
    conv4 = maxpool2d(conv4)
    #Convloution layer 5
    conv5 = conv2d(conv4, weight_list[4], bias_list[4])
    conv5 = maxpool2d(conv5)
    #Fully connected layer
    # Reshapes last layer accordingly

    conv5_flattened = tf.reshape(conv5,[-1, weight_list[5].shape[0]])
    #Multiplies fc1 and 'wd1' and then adds it with bias
    fc1 = tf.add(tf.matmul(conv5_flattened, weight_list[5]),bias_list[5])
    #Applies relu
    fc1 = tf.nn.relu(fc1)

    #Output layer
    out = tf.add(tf.matmul(fc1,weight_list[6]), bias_list[6])
    #print (out.shape)
    return out

# Predictions are  stored here
pred = conv_net(x, weight_list, bias_list)

# Cost is the 'average' of the loss

loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(loss)

#Optimazion method

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
#for frame in range(3600,6000):
 #   z = random.randint(2000,156000) #should be ay
 #   while z%6000<100 or z%6000>5900 or full_script[z][0] =='not-found-in-audio' or full_script[z][0] =='oov':

 #       z = random.randint(0,156000)
z = 224
img = framenumToimg(z)/255
batch_x = img
batch_x = np.reshape(batch_x, (1,385,413,3))
if full_script[z-1][0][-2] == '_':
    _y = (Phonemes[full_script[z-1][0][:-2]])
else:
    _y = (Phonemes[full_script[z-1][0]])
#print ('using frame ' + str(z) + ' and the label is ' + str(fake_batch_y[-1]))

#print ('created batch_x and _y')

#batch_x is defined as a numpy array of fake_batch_x


#print (np.arange(batch_size))
#print (fake_batch_y)

#print ('running prediction')
prediction = sess.run(pred, feed_dict = {x:batch_x})
correct_prediction = tf.argmax(prediction, 1)
print ('\n')
print (z)
print (sess.run(correct_prediction))
print (_y)

print (sess.run(correct_prediction) == _y)