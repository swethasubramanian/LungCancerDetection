from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import numpy as np

import pickle
from CreateImages import CTScan


X_train = pd.read_pickle('traindata')

def create_data(idx, width = 50):
	scan = CTScan(np.asarray(X_train.loc[idx])[0], \
		np.asarray(X_train.loc[idx])[1:])
	outfile = 'train/image_'  +  str(idx)+ '.tiff'
	scan.save_image(outfile, width)


Parallel(n_jobs = 3)(delayed(create_data)(idx) for idx in X_train.index)