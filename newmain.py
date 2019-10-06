#newmain.py
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


#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
#Imports data from tensorflow library 'input_data'. one_hot transforms categorical labels into binary vectors
'''one_hot - In one-hot encoding, you convert the categorical data into a vector of numbers. You do this because machine learning algorithms can't work with categorical data directly. 
Instead, you generate one boolean column for each category or class. Only one of these columns could take on the value 1 for each sample. That explains the term "one-hot encoding".'''
data = input_data.read_data_sets('data/fashion', one_hot = True)

#All images are scaled between 0-1 in this dataset. In artifical dataset we need to scale it down to this value

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

#New line
print ('\n')

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))

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


#Only works with %matplotlib inline - requires IPython
#Displays 2 images
'''plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")'''

# Gives matrix version of image '0'
print(data.train.images[0])
# Prints largest value, should be 1
print(np.max(data.train.images[0]))

#Prints smallest value, should be 0
print(np.min(data.train.images[0]))

