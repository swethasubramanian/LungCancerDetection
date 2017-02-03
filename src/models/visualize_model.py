"""
A script to visualize layers in the conv net model 
"""

import tflearn
from cnn_model import CNNModel

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread


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

def plot_layers(layer, model, inp, idx):
	"""
	plot filter output in layers

	Args
	----
	layer: cnn layer
	model: cnn model
	inp: input image
	idx: layer number
	"""
	m2 = tflearn.DNN(layer, session=model.session)
	yhat = m2.predict(inp.reshape(-1, inp.shape[0], inp.shape[1], 1))
	yhat_1 = np.array(yhat[0])

	print yhat_1.shape

	
	nfilt = yhat_1.shape[2]

	mosaic = create_mosaic(yhat_1, 4, nfilt/4)
	plt.figure(figsize = (12,12))
	plt.imshow(mosaic, cmap = 'magma')
	plt.axis('off')
	plt.savefig('conv_layer_' + str(idx)+'.png', bbox_inches='tight')
	#plt.show()

### Plot layer
filename = '../data/test/image_315896.jpg'
inp = imread(filename).astype('float32')


convnet  = CNNModel()
inp_layer, conv_layer_1, mp_layer_1, conv_layer_2, conv_layer_3, mp_layer_2,\
fully_connected_layer_1, dropout_layer_1, softmax_layer, network =\
 convnet.define_network(inp.reshape(-1, inp.shape[0], inp.shape[1], 1), 'visual')
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule3-classifier.tfl.ckpt')
model.load("nodule2-classifier.tfl")

layers_to_be_plotted = [conv_layer_1, conv_layer_2, conv_layer_3]
#plot_layers(conv_layer_1, model, inp)
for idx, layer in enumerate(layers_to_be_plotted):
	plot_layers(layer, model, inp, idx)






