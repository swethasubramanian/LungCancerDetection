"""
A script to visualize layers in the conv net model and for analysis of results
"""

import tflearn
from cnn_model import CNNModel

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py

import matplotlib.pyplot as plt

h5f2 = h5py.File('../data/test.h5', 'r')
X_test_images = h5f2['X']
Y_test_labels = h5f2['Y']

## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_test_images)
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule3-classifier.tfl.ckpt')
model.load("nodule3-classifier.tfl")

print tflearn.variables.get_()

