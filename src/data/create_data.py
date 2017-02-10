#!/usr/bin/env python

import sys

from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import numpy as np

import pickle
from create_images import CTScan


if len(sys.argv) < 2:
	raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
else:
	mode = sys.argv[1]
	if mode not in ['train', 'test', 'val']:
		raise ValueError('Argument not recognized. Has to be train, test or val')

inpfile = mode + 'data'
outDir = mode + '/image_'
X_data = pd.read_pickle(inpfile)
raw_image_path = '../../data/raw/*/'

def create_data(idx, outDir, width = 50):
	'''
	Generates your test, train, validation images
	outDir = a string representing destination
	width (int) specify image size
	'''
	scan = CTScan(np.asarray(X_data.loc[idx])[0], \
		np.asarray(X_data.loc[idx])[1:], raw_image_path)
	outfile = outDir  +  str(idx)+ '.jpg'
	scan.save_image(outfile, width)


# Parallelizes inorder to generate more than one image at a time
Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir) for idx in X_data.index)
