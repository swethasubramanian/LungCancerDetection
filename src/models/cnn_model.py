"""
A conv net model using tflearn wrapper for tensorflow
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py


class CNNModel(object):
	
	def __init__(self, network = None):
		self.network = network
		self.model = None

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


	def input_layer(self, X_images, name):
		"""
		Define Input layer
		"""
		img_prep = self.preprocessing()
		img_aug = self.augmentation()
		self.network = input_data(shape = [None, X_images.shape[1], X_images.shape[2], X_images.shape[3]],
                     data_preprocessing = img_prep,
                     data_augmentation = img_aug,
                     name = name)


	def convolution_layer(self, num_filters, filter_size, name, activation_type = 'relu', regularizer = 'L1'):
		"""
		Creates a 2D-conv layer

		Args:
		-----
		num_filters = takes an integer
		filter_size = takes an integer
		activation = takes a string
		"""
		self.network = conv_2d(self.network, num_filters,\
		 filter_size, activation = activation_type, regularizer = regularizer, name = name)

	def max_pooling_layer(self, kernel_size, name):
		"""
		It is common to periodically insert a Pooling layer in-between successive Conv layers 
		in a ConvNet architecture. Its function is to progressively reduce the spatial size of
		the representation to reduce the amount of parameters and computation in the 
		network, and hence to also control overfitting. 

		args:
		-----
		kernel_size: takes an integer
		"""
		self.network = max_pool_2d(self.network, kernel_size, name = name)

	def fully_connected_layer(self, num_units, activation_type, name):
		"""
		Neurons in a fully connected layer have full connections to all activations in the previous
		layer, as seen in regular Neural Networks. Their activations can hence be computed with
		 a matrix multiplication followed by a bias offset. 

		 args:
		 ------
		 num_units: an integer representing number of units in the layer

		"""
		self.network = fully_connected(self.network, num_units, activation= activation_type, name = name)

	def dropout_layer(self, name, prob = 0.5):
		"""
		args
		------
		prob = float representing dropout probability

		"""
		if (prob > 1) or (prob < 0):
			raise ValueError('Probability values should e between 0 and 1')
		self.network = dropout(self.network, prob, name = name)

	def define_network2(self, X_images):
		self.network = input_data(shape=[None, 50, 50, 1], name='input')
		self.network = conv_2d(self.network, 50, 3, activation='relu', regularizer="L2")
		self.network = max_pool_2d(self.network, 2)
		self.network = local_response_normalization(self.network)
		self.network = conv_2d(self.network, 64, 3, activation='relu', regularizer="L2")
		self.network = max_pool_2d(self.network, 2)
		self.network = local_response_normalization(self.network)
		self.network = fully_connected(self.network, 128, activation='tanh')
		self.network = dropout(self.network, 0.8)
		self.network = fully_connected(self.network, 256, activation='tanh')
		self.network = dropout(self.network, 0.8)
		self.network = fully_connected(self.network, 10, activation='softmax')
		self.network = regression(self.network, optimizer='adam', learning_rate=0.01,
			loss='categorical_crossentropy', name='target')
		return self.network



	def define_network1(self, X_images):
		"""
		Creates a regression network
		Args:
		-------
		X_images: A HDF5 datasets representing input layer

		"""
		self.input_layer(X_images, name = 'inp1')
		self.convolution_layer(64, 3, 'conv1', 'relu') # 50 filters, with size 3
		self.max_pooling_layer(2, 'mp1') # downsamples spatial size by 2
		self.convolution_layer(128, 3, 'conv2', 'relu')
		self.convolution_layer(256, 3, 'conv3', 'relu')
		self.max_pooling_layer(2, 'mp2')
		self.fully_connected_layer(512,'relu', 'fl1')
		self.dropout_layer('dp1', 0.5)
		self.fully_connected_layer(2, 'softmax', 'fl2')

		self.network = regression(self.network, optimizer = 'adam',\
		 loss = 'categorical_crossentropy', learning_rate = 0.001)

		return self.network

	def define_network(self, X_images):
		"""
		Creates a regression network
		Args:
		-------
		X_images: A HDF5 datasets representing input layer

		"""
		self.input_layer(X_images)
		self.convolution_layer(50, 3, 'relu') # 50 filters, with size 3
		self.max_pooling_layer(2) # downsamples spatial size by 2
		self.convolution_layer(64,3,'relu')
		self.convolution_layer(64,3, 'relu')
		self.max_pooling_layer(2)
		self.fully_connected_layer(512, 'relu')
		self.dropout_layer('dp1', 0.5)
		self.fully_connected_layer(2, 'softmax')

		self.network = regression(self.network, optimizer = 'adam',\
		 loss = 'categorical_crossentropy', learning_rate = 0.001)

		return self.network


	