LungCancerProject
==============================

Creating a model to detect lung nodules using CNN from LIDR data

## Project Organization
------------
### Analysis Notebooks
LungCancerDetection.ipynb -> Analysis notebook

### Data Processing Scripts:
src/data/

1. augment_images.py: script to augment training set
2. build_hdf5_datasets.py: script that concatenates all the images to form a 3D volume. Takes 'train', 'test', 'val' as arguments to generate training set, test set and a validation set.
3. create_data.py: Generates cropped test, train and validation images. Takes 'train', 'test', 'val' as arguments to generate training, test and a validation images.
4. create_images.py: A class that contains methods to process and extract images from .mhd data
5. test_train_split.py: Reads csv files and random splits data into test train and validation

### Models
src/model/

1. train_model.py: A script to train CNN model
  
