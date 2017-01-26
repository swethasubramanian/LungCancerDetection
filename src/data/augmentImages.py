## A script to augment minority class

from scipy.ndimage import rotate, imread
from PIL import Image

from joblib import Parallel, delayed
import multiprocessing

import pickle

with open ('posfile', 'rb') as fp:
    itemlist = pickle.load(fp)

def augment_positive_cases(idx):
	inp = imread('train/image_' + str(idx)+ '.tiff')
	# Rotate by 90
	inp90 = rotate(inp, 90, reshape = False)
	Image.fromarray(inp90).convert('L').save('train/' +\
		'image_' + str(idx+1000000) + '.tiff')

Parallel(n_jobs = 2)(delayed(augment_positive_cases)(idx) for idx in itemlist)





