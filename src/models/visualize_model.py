"""
A script to visualize layers in the conv net model and for analysis of results
"""

"""
A script to visualize CNN model
"""

import tflearn
from cnn_model import CNNModel

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread


def create_mosaic(image, nrows, ncols):
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

def plot_layers(layer):
	m2 = tflearn.DNN(layer, session=model.session)
	yhat = m2.predict(inp.reshape(-1, inp.shape[0], inp.shape[1], 1))
	yhat_1 = np.array(yhat[0])

	mosaic = create_mosaic(yhat_1, 2, 16)
	plt.figure(figsize = (12,12))
	plt.imshow(mosaic, cmap = 'gray')
	plt.axis('off')
	plt.savefig(str(layer)+'.png', bbox_inches='tight')
	plt.show()

def 


## Model definition
convnet  = CNNModel()
inp_layer, conv_layer_1, mp_layer_1, conv_layer_2, conv_layer_3, mp_layer_2,\
fully_connected_layer_1, dropout_layer_1, softmax_layer, network =\
 convnet.define_network(X_test_images, 'visual')
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule3-classifier.tfl.ckpt')
model.load("nodule2-classifier.tfl")

### Plot the input layer
inp = imread('../data/test/image_100160.jpg').astype('float32')
print inp.shape
# conv_layer_variables = tflearn.variables.get_layer_variables_by_name('conv1')
# weights_conv1_1 = model.get_weights(conv_layer_variables[0])
# weights_conv1_2 = model.get_weights(conv_layer_variables[1])
# print weights_conv1_1.shape, weights_conv1_2.shape
#plt.show()

## First layer











