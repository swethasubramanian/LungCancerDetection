"""
A script to visualize layers and filters in the conv net model 
"""

import tflearn
from cnn_model import CNNModel

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread, zoom

def create_mosaic(image, nrows, ncols):
	"""
	Tiles all the layers in nrows x ncols
	Args:
	------
	image = 3d numpy array of M * N * number of filters dimensions
	nrows = integer representing number of images in a row
	ncol = integer representing number of images in a column

	returns formatted image
	"""
	M = image.shape[0]
	N = image.shape[1]

	npad = ((1,1), (1,1), (0,0))
	image = np.pad(image, pad_width = npad, mode = 'constant',\
		constant_values = 0)
	M += 2
	N += 2
	image = image.reshape(M, N, nrows, ncols)
	image = np.transpose(image, (2,0,3,1))
	image = image.reshape(M*nrows, N*ncols)
	return image

def get_layer_output(layer, model, inp):
	"""
	Returns model layer output

	Args
	----
	layer: cnn layer
	model: cnn model
	inp: input image

	"""
	m2 = tflearn.DNN(layer, session = model.session)
	yhat = m2.predict(inp.reshape(-1, inp.shape[0], inp.shape[1], 1))
	yhat_1 = np.array(yhat[0])
	return m2, yhat_1


def plot_layers(image, idx, pltfilename, size = 12, cmapx = 'magma'):
	"""
	plot filter output in layers

	Args
	----
	image: layer output of form M x N x nfilt
	idx: layer number
	pltfilename = a string representing filename

	"""
	nfilt = image.shape[-1]

	mosaic = create_mosaic(image, nfilt/4, 4)
	plt.figure(figsize = (size, size))
	plt.imshow(mosaic, cmap = cmapx)
	plt.axis('off')
	plt.savefig(pltfilename + str(idx)+'.png', bbox_inches='tight')
	#plt.show()

def get_weights(m2, layer):
	"""
	get a layer's weights

	Args:
	------
	m2: model input
	layer = layer in question

	Returns:
	weights 
	"""
	weights = m2.get_weights(layer.W)
	print weights.shape
	weights =\
	 weights.reshape(weights.shape[0], weights.shape[1], weights.shape[-1])
	return weights

def plot_single_output(image, size = 6):
	plt.figure(figsize = (size, size))
	plt.imshow(mosaic, cmap = 'magma')
	plt.axis('off')
	plt.savefig('filterout' + '.png', bbox_inches='tight')	



def main():
	### Plot layer
	filename = '../data/test/image_21351.jpg'
	inp = imread(filename).astype('float32')


	convnet  = CNNModel()
	conv_layer_1, conv_layer_2, conv_layer_3, network =\
	 convnet.define_network(inp.reshape(-1, inp.shape[0], inp.shape[1], 1), 'visual')
	model = tflearn.DNN(network, tensorboard_verbose=0,\
			 checkpoint_path='nodule3-classifier.tfl.ckpt')
	model.load("nodule3-classifier.tfl")
	print model.predict(inp.reshape(-1, 50, 50, 1))



	layers_to_be_plotted = [conv_layer_1, conv_layer_2, conv_layer_3]
	#plot_layers(conv_layer_1, model, inp)
	for idx, layer in enumerate(layers_to_be_plotted):
		m2, yhat = get_layer_output(layer, model, inp)
		plot_layers(yhat, idx, 'conv_layer_')

	weights = get_weights(m2, conv_layer_1)
	plot_layers(weights, 0, 'weight_conv_layer_', 6, 'gray')

if __name__ == "__main__":
	main()







