
import tflearn
import h5py
import numpy as np
from cnn_model import CNNModel 


# Load HDF5 dataset
h5f = h5py.File('../data/train.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']


h5f2 = h5py.File('../data/val.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']


## Model definition
convnet  = CNNModel()
convnet.define_network(X_train_images)
convnet.define_model()
convnet.train_model(X_train_images, Y_train_labels, X_val_images, Y_val_labels)


# Save model when training is complete to a file
model.save("nodule-classifier.tfl")
print("Network trained and saved as nodule-classifier.tfl!")

h5f.close()
h5f2.close()