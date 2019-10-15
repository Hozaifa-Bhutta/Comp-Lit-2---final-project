#test_feed.py
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
from reads_csv import read_file
import random
x = tf.placeholder('float', [None, 480, 480, 1])
y = tf.placeholder('float', [None, 2])
file = read_file('full.csv', 10)
file = np.reshape(file,[-1,480,480,1])
file2 = read_file('full.csv', 12)
file2 = np.reshape(file,[-1,480,480,1])
batch_x = np.concatenate((file,file2))

sess = tf.Session()
z = x*2

print (batch_x.shape)

sess.run(z, feed_dict={x:file})
