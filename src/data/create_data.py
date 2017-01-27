#!/usr/bin/env python

import sys

from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import numpy as np

import pickle
from CreateImages import CTScan


def create_data(idx, inpfile, outDir, width = 50):
	scan = CTScan(np.asarray(X_data.loc[idx])[0], \
		np.asarray(X_data.loc[idx])[1:])
	outfile = outDir  +  str(idx)+ '.tiff'
	scan.save_image(outfile, width)

mode = sys.argv[1]
inpfile = mode + 'data'
outDir = mode + '/image_'
X_data = pd.read_pickle(inpfile)

Parallel(n_jobs = 3)(delayed(create_data)(idx, inpfile, outDir) for idx in X_data.index)
