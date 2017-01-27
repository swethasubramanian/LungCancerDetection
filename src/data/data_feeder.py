import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import pandas as pandas


class InputData(object):
	'''
	Creates a queue for loading and feeding images to CNN model in batches
	'''
    def __init__(self, X = None, y = None, filepath = None):
        self.X = X
        self.y = y
        self.image = None
        self.label = None
        self.filepath = filepath
    
    def get_filenames_labels(self):
        filenames =\
        self.X.index.to_series().apply(lambda x:\
        	self.filepath + str(x) + '.jpg')
        images = ops.convert_to_tensor(filenames.values, dtype=dtypes.string)
        labels = ops.convert_to_tensor(self.y.values.astype(int), dtype=dtypes.int64)
        return images, labels
    
    def get_images_labels(self):
        return self.image, self.label
    
    def build_input_queues(self):
        '''
        Builds an input queue and defines how to load these images
        '''
        images, labels = self.get_filenames_labels()
        input_queue = tf.train.slice_input_producer([images, labels],\
        	shuffle = False)

        file_content = tf.read_file(input_queue[0])
        self.image = tf.image.decode_jpeg(file_content, channels = 1)
        self.label = input_queue[1]
        return self.image, self.label 
    
    def create_image_batch(self, img_ht, img_width, num_channels, BATCH_SIZE):
        self.image.set_shape([img_ht, img_width, num_channels])
        image_batch, label_batch = tf.train.batch([self.image, self.label], \
        	batch_size = BATCH_SIZE)
        return image_batch, label_batch 









