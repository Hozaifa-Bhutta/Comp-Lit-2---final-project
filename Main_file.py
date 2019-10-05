import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math
import os
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
y_train /= 255
y_test /= 255 

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
#convolution and maxpool laters
conv1 = tf.layers.conv2d(inputs=x_train, filters=28, kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu)
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=(2,2), padding='same')
print (maxpool1.get_shape())
#flattens maxpool1
maxpool1_flat = tf.reshape(maxpool1, [-1,14*14*28])
print (maxpool1_flat.get_shape())

#Fully connected layers
hidden = tf.layers.dense(inputs=maxpool1_flat, units=100, activation=tf.nn.relu)
hidden2 = tf.layers.dense(inputs=hidden, units=10, activation=tf.nn.softmax)
print (hidden2.get_shape())

#make y_train compataible with sparse_softmax_cross_entropy
y_train=y_train.astype('int32')

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_train, logits=hidden2)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.0002).minimize(cost)
sess=tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
train_loss, _, _logits = sess.run([cost, opt, hidden2])
