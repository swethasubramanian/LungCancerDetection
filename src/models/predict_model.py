"""
A script to train a conv net model using tflearn wrapper for tensorflow
"""

import tflearn
from cnn_model import CNNModel

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt



h5f2 = h5py.File('../data/test.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_test_images)
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule-classifier.tfl.ckpt')
model.load("nodule-classifier.tfl")

#convnet.define_model()
#predictions, score = convnet.predict_results(X_test_images, Y_test_labels)

predictions = np.vstack(model.predict(X_test_images[:,:,:,:]))
label_predictions = np.vstack(model.predict_label(X_test_images[:,:,:,:]))
score = model.evaluate(X_test_images, Y_test_labels)

## ROC
fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

precision = TP*1.0/(TP+FP)
recall = TP*1.0/(TP+FN)
specificity = TN*1.0/(TN+FP)

print precision, recall, specificity
print TP, FP, FN, TN


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
plt.axis('tight')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc.png', bbox_inches='tight')
plt.show()



