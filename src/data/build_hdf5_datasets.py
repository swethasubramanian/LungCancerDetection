#!/usr/bin/env python


import sys

import numpy as np 
import pandas as pd 

from tflearn.data_utils import build_hdf5_image_dataset

import pickle

mode = sys.argv[1]

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

build_hdf5_image_dataset(dataset_file, image_shape=(50, 50),\
 mode='file', output_path=output, categorical_labels=True, normalize=True)






