import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import pandas as pandas
import pickle


## Format training, test and validation datasets so that we can batch feed them to CNN models 
class Dataset(object):
	def __init__(self, X = None, y = None, filepath = None):
		self.X = X
		self.y = y
		self.image = None
		self.label = None
		self.filepath = filepath

	def get_filenames_labels(self):
		filenames =\
		self.X.index.to_series().apply(lambda x:\
			self.filepath + str(x) + '.tiff').to_string()
		images = ops.convert_to_tensor(filenames, dtype=dtypes.string)
		labels = ops.convert_to_tensor(y.values, dtype=dtypes.float64)
		return images, labels

	def get_images_labels(self):
		return self.image, self.label

	def build_input_queues(self):
		'''
		Builds an input queue and defines how to load these images
		'''
		images, labels = self.get_filenames_labels()
		input_queue =\
		 tf.train.slice_input_producer([images, labels],\
		 shuffle = False)

		file_content = tf.read_file(input_queue[0])
		self.image = tf.image.decode_jpeg(file_content, channels = 1)
		self.label = input_queue[1]
		return self.image, self.label 

	def define_tensor_shape(self, img_ht, img_width, num_channels):
		self.image.set_shape([img_ht, img_width, num_channels])

	def create


X_train = pd.read_pickle('traindata')
y_train = pd.read_pickle('trainlabels')

X_test = pd.read_pickle('testdata')
y_test = pd.read_pickle('testlabels')

X_val = pd.read_pickle('valdata')
y_val = pd.read_pickle('vallabels')


# Start with the augmented trained dataset
tempDf = X_train[y_train == 1]
tempDf = tempDf.set_index(X_train[y_train == 1].index + 1000000)
X_train_new = X_train.append(tempDf)
tempDf = tempDf.set_index(X_train[y_train == 1].index + 2000000)
X_train_new = X_train_new.append(tempDf)

ytemp = y_train.reindex(X_train[y_train == 1].index + 1000000)
ytemp.loc[:] = 1
y_train_new = y_train.append(ytemp)
ytemp = y_train.reindex(X_train[y_train == 1].index + 2000000)
ytemp.loc[:] = 1
y_train_new = y_train_new.append(ytemp)

train_images, train_labels = get_filenames_labels(X_train_new, y_train_new, 'train/images_')
test_images, test_labels = get_filenames_labels(X_test, y_test, 'test/images_')
val_images, val_labels = get_filenames_labels(X_test, y_test, 'test/images_')




