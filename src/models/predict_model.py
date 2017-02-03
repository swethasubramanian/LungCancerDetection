"""
A script to predict nodules using conv net model and for analysis of results
"""

import tflearn
from cnn_model import CNNModel

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.grid('off')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


h5f2 = h5py.File('../data/test.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_test_images)
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule3-classifier.tfl.ckpt')
model.load("nodule2-classifier.tfl")

#convnet.define_model()
#predictions, score = convnet.predict_results(X_test_images, Y_test_labels)

predictions = np.vstack(model.predict(X_test_images[:,:,:,:]))
#label_predictions = np.vstack(model.predict_label(X_test_images[:,:,:,:]))
score = model.evaluate(X_test_images, Y_test_labels)
label_predictions = np.zeros_like(predictions)
label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1

## ROC
fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])
print Y_test_labels[:,1].sum(), label_predictions[:,1].sum()
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
         lw=lw, label='(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.axis('tight')
plt.axis('equal')
#plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc1.png', bbox_inches='tight')


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['no-nodule', 'nodule'],
                      title='Confusion matrix')
plt.savefig('confusion_matrix.png', bbox_inches='tight')

plt.show()







