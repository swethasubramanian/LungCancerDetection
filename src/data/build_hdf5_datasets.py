#!/usr/bin/env python
"""
Builds a HDF5 data set for test, train and validation data
Run script as python build_hdf5_datasets.py $mode
where mode can be 'test', 'train', 'val'
"""

import sys
import numpy as np 
import pandas as pd 
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import pickle
import h5py


# Check inputs
if len(sys.argv) < 2:
	raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
else:
	mode = sys.argv[1]
	if mode not in ['train', 'test', 'val']:
		raise ValueError('Argument not recognized. Has to be train, test or val')

# Read data
X = pd.read_pickle(mode + 'data')
y = pd.read_pickle(mode + 'labels')


dataset_file = mode + 'datalabels.txt'

filenames =\
X.index.to_series().apply(lambda x:\
	mode+ '/image_'+str(x)+'.jpg')

   
filenames = filenames.values.astype(str)
labels = y.values.astype(int)
data = np.zeros(filenames.size,\
                     dtype=[('var1', 'S36'), ('var2', int)])
data['var1'] = filenames
data['var2'] = labels

np.savetxt(dataset_file, data, fmt="%10s %d")

output = mode + 'dataset.h5'

build_hdf5_image_dataset(dataset_file, image_shape = (50, 50, 1), \
 mode ='file', output_path = output, categorical_labels = True, normalize = True,
 grayscale = True)

# Load HDF5 dataset
h5f = h5py.File('../data/'+ mode+ 'dataset.h5', 'r')
X_images = h5f['X']
Y_labels = h5f['Y'][:]

print X_images.shape
X_images = X_images[:,:,:].reshape([-1,50,50,1])
print X_images.shape
h5f.close()

h5f = h5py.File('../data/' + mode + '.h5', 'w')
h5f.create_dataset('X', data=X_images)
h5f.create_dataset('Y', data=Y_labels)
h5f.close()








