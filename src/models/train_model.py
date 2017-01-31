"""
A script to train a conv net model using tflearn wrapper for tensorflow
"""


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py



# Load HDF5 dataset
h5f = h5py.File('../data/traindataset.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']
print X_train_images.shape

h5f2 = h5py.File('../data/valdataset.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']

h5f2 = h5py.File('../data/testdataset.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

# class Model(object):
	
# 	def __init__(self, network = None):
# 		self.network = network

# 	def preprocessing(self):
# 		"""
# 		Make sure the data is normalized
# 		"""
# 		img_prep = ImagePreprocessing()
# 		img_prep.add_featurewise_zero_center()
# 		img_prep.add_featurewise_stdnorm()
# 		return img_prep

# 	def augmentation(self):
# 		"""
# 		Create extra synthetic training data by flipping, rotating and blurring the
# 		images on our data set.
# 		"""
# 		img_aug = ImageAugmentation()
# 		img_aug.add_random_flip_leftright()
# 		img_aug.add_random_rotation(max_angle=25.)
# 		img_aug.add_random_blur(sigma_max=3.)
# 		return img_aug


# 	def input_layer(self, X_images):
# 		"""
# 		Define Input layer
# 		"""
# 		self.network = input_data(shape = [None, 50, 50],
#                      data_preprocessing = img_prep,
#                      data_augmentation = img_aug)





# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Input is a 50x50 image with 1 color channels (grayscale)
#X_train_images = tf.reshape(X_train_images, [-1, 50, 50, 3])

network = input_data(shape = [None, 50, 50, 3],
                     data_preprocessing = img_prep,
                     data_augmentation = img_aug)

# Step 1: Convolution
network = conv_2d(network, 50, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a nodule, 1=is a nodule) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X_train_images, Y_train_labels, n_epoch=50, shuffle=True,\
 validation_set = (X_val_images, Y_val_labels), show_metric = True,\
  batch_size = 96, snapshot_epoch = True, run_id = 'nodule-classifier')

# Save model when training is complete to a file
model.save("nodule-classifier.tfl")
print("Network trained and saved as nodule-classifier.tfl!")



