from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import pickle
import CreateImages


X_test = pd.read_pickle('testdata')

def create_data(idx, width = 50):
	scan = CTScan(np.asarray(X_test.loc[idx])[0], \
		np.asarray(X_test.loc[idx])[1:])
	outfile = 'test/image_'  +  str(idx)+ '.tiff'
	scan.save_image(outfile, width)


Parallel(n_jobs = 3)(delayed(create_data)(idx) for idx in X_test.index)