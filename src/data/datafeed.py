import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import pandas as pandas
import pickle


## Format training, test and validation datasets so that we can batch feed them to CNN models 

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

print len(X_train_new), len(y_train_new)

train_filenames =\
X_train_new.index.to_series().apply(lambda x:\
                                    '../src/data/train/images_'+str(x)+'.tiff').to_string()