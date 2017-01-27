LungCancerProject
==============================

Creating a model to detect lung nodules using CNN from LIDR data

Project Organization
------------
Data Processing Scripts:
|
|_notebooks/
    |__LungCancerDetection.ipynb -> Analysis notebook
    |
|_src/data/:
    |__ augment_images.py      -> script to augment training set
    |
    |__ build_hdf5_datasets.py -> script that concatenates all the images to form a 3D volume. Takes 'train', 'test', 'val' as arguments to generate training set, test set and a validation set.
    |
    |_create_data.py           -> Generates cropped test, train and validation images. Takes 'train', 'test', 'val' as arguments to generate training, test and a validation images.
    |
    |__create_images.py        -> A class that contains methods to process and extract images from .mhd data
    |
    |__test_train_split.py     -> Reads csv files and random splits data into test train annd validation
  



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
