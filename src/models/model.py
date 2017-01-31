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

class Model(object):
	
	def __init__(self, network = None):
		self.network = network

	def preprocessing(self):
		"""
		Make sure the data is normalized
		"""
		img_prep = ImagePreprocessing()
		img_prep.add_featurewise_zero_center()
		img_prep.add_featurewise_stdnorm()
		return img_prep

	def augmentation(self):
		"""
		Create extra synthetic training data by flipping, rotating and blurring the
		images on our data set.
		"""
		img_aug = ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_rotation(max_angle=25.)
		img_aug.add_random_blur(sigma_max=3.)
		return img_aug


	def input_layer(self, X_images):
		"""
		Define Input layer
		"""
		self.network = input_data(shape = [None, X_images.shape[1], X_images.shape[2], X_images.shape[3]],
                     data_preprocessing = img_prep,
                     data_augmentation = img_aug)


	def convolution_layer(self, num_filters, filter_size, activation_type = 'relu'):
		"""
		Creates a 2D-conv layer

		Args:
		-----
		num_filters = takes an integer
		filter_size = takes an integer
		activation = takes a string
		"""
		self.network = conv_2d(self.network, num_filters, filter_size, activation_type)

	def max_pooling_layer(self, kernel_size):
		"""
		It is common to periodically insert a Pooling layer in-between successive Conv layers 
		in a ConvNet architecture. Its function is to progressively reduce the spatial size of
		the representation to reduce the amount of parameters and computation in the 
		network, and hence to also control overfitting. 

		args:
		-----
		kernel_size: takes an integer
		"""
		self.network = max_pool_2d(self.network, kernel_size)

	def fully_connected_layer(self, num_units):
		"""
		Neurons in a fully connected layer have full connections to all activations in the previous
		layer, as seen in regular Neural Networks. Their activations can hence be computed with
		 a matrix multiplication followed by a bias offset. 

		 args:
		 ------
		 num_units: an integer representing number of units in the layer

		"""
		self.network = fully_connected(self.network, 512, activation='relu')

	def dropout_layer(self, prob = 0.5):
		"""
		args
		------
		prob = float representing dropout probability

		"""
		if (prob > 1) or (prob < 0):
			raise ValueError('Probability values should e between 0 and 1')

		self.network = dropout(self.network, prob)



	def define_network(self):
		"""

		"""


