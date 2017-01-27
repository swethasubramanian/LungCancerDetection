
import tensorflow as tf
import pickle

from datafeed import Dataset

import pandas as pd

class LungModel(object):
	'''
	Will initialize model here
	'''

	def __init__(self):
		

## Feed trining data into tensorflow in batches
X_train = pd.read_pickle('traindata')
y_train = pd.read_pickle('trainlabels')

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


