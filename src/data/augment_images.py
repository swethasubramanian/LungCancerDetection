"""
 A script to augment minority class in the training set, by flipping images
"""

from scipy.ndimage import rotate, imread
from PIL import Image

import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

import pickle

X_train = pd.read_pickle('traindata')
y_train = pd.read_pickle('trainlabels')

augIndexes = X_train[y_train == 1].index


def augment_positive_cases(idx):
	inp = imread('train/image_' + str(idx)+ '.jpg')
	# Rotate by 90
	inp90 = rotate(inp, 90, reshape = False)
	Image.fromarray(inp90).convert('L').save('train/' +\
		'image_' + str(idx+1000000) + '.jpg')

	inp180 = rotate(inp, 180, reshape = False)
	Image.fromarray(inp180).convert('L').save('train/' +\
		'image_' + str(idx+2000000) + '.jpg')

Parallel(n_jobs = 3)(delayed(augment_positive_cases)(idx) for idx in augIndexes)





