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
	"""
	Initializes a convolution neural network model for training, prediction, and visualization
	"""
	
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
		return self.network


	def convolution_layer(self, num_filters, filter_size, name, activation_type = 'relu', regularizer = None):
		"""
		Creates a 2D-conv layer

		Args:
		-----
		num_filters = takes an integer
		filter_size = takes an integer
		name = takes a string 
		activation = takes a string
		regularizer = 'L1' or 'L2' or None
		"""
		self.network = conv_2d(self.network, num_filters,\
		 filter_size, activation = activation_type, regularizer = regularizer, name = name)
		return self.network

	def max_pooling_layer(self, kernel_size, name):
		"""
		It is common to periodically insert a Pooling layer in-between successive Conv layers 
		in a ConvNet architecture. Its function is to progressively reduce the spatial size of
		the representation to reduce the amount of parameters and computation in the 
		network, and hence to also control overfitting. 

		args:
		-----
		kernel_size: takes an integer
		name : a str representing name of the layer
		"""
		self.network = max_pool_2d(self.network, kernel_size, name = name)
		return self.network

	def fully_connected_layer(self, num_units, activation_type, name):
		"""
		Neurons in a fully connected layer have full connections to all activations in the previous
		layer, as seen in regular Neural Networks. Their activations can hence be computed with
		 a matrix multiplication followed by a bias offset. 

		 args:
		 ------
		 num_units: an integer representing number of units in the layer

		"""
		self.network = fully_connected(self.network, num_units,\
		 activation= activation_type, name = name)
		return self.network

	def dropout_layer(self, name, prob = 0.5):
		"""
		args
		------
		prob = float representing dropout probability

		"""
		if (prob > 1) or (prob < 0):
			raise ValueError('Probability values should e between 0 and 1')
		self.network = dropout(self.network, prob, name = name)
		return self.network


	def define_network(self, X_images, mode = 'testtrain'):
		"""
		Creates a regression network
		Args:
		-------
		X_images: A HDF5 datasets representing input layer

		Returns
		A CNN network

		if mode is visual: then it returns intermediate layers as well

		"""
		inp_layer = self.input_layer(X_images, name = 'inpu1')
		conv_layer_1 = self.convolution_layer(32, 5, 'conv1', 'relu', 'L2') # 50 filters, with size 3
		mp_layer_1 = self.max_pooling_layer(2, 'mp1') # downsamples spatial size by 2
		conv_layer_2 = self.convolution_layer(64, 5, 'conv2', 'relu', 'L2')
		conv_layer_3 = self.convolution_layer(64, 3, 'conv3', 'relu', 'L2')
		mp_layer_2 = self.max_pooling_layer(2, 'mp2')
		fully_connected_layer_1 = self.fully_connected_layer(512,'relu', 'fl1')
		dropout_layer_1 = self.dropout_layer('dp1', 0.5)
		softmax_layer  = self.fully_connected_layer(2, 'softmax', 'fl2')

		self.network = regression(self.network, optimizer = 'adam',\
		 loss = 'categorical_crossentropy', learning_rate = 0.001)
		
		if mode == 'testtrain':
			return self.network
		if mode == 'visual':
			return conv_layer_1, conv_layer_2, conv_layer_3, self.network







	