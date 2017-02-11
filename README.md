LungCancerProject
==============================


Deep learning is a fast and evolving field that has a lot of implications on medical imaging field.
 
Currently medical images are interpreted by radiologists, physicians etc. But this interpretation gets very subjective. After years of looking at ultrasound images, my co-workers and I still get into arguments about whether we are actually seeing a tumor in a scan. Radiologists often have to look through large volumes of these images that can cause fatigue and lead to mistakes. So there is a need for automating this.

Machine learning algorithms such as support vector machines are often used to detect and classify tumors. But they are often limited by the assumptions we make when we define features. This results in reduced sensitivity.
However, deep learning could be ideal solution because these algorithms are able to learn features from raw image data.

One challenge in implementing these algorithms is the scarcity of labeled medical image data. While this is a limitation for all applications of deep learning, it is more so for medical image data because of patient confidentiality concerns.

In this post you  will learn how to build a convolutional neural network, train it, and have it detect lung nodules. I used the data from the Lung Image Database Consortium and Infectious Disease Research Institute [(LIDC/IDRI) data base] (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). As these images were huge (124 GB), I ended up using reformatted version available for [LUNA16](https://luna16.grand-challenge.org/data/). This dataset consisted of 888 CT scans with annotations describing coordinates and ground truth labels. First step was to create a image database for training.

### Creating an image database

The images were formatted as .mhd and .raw files. The header data is contained in .mhd files and multidimensional image data is stored in .raw files. I used [SimpleITK](http://www.simpleitk.org/) library to read the .mhd files. Each CT scan has dimensions of 512 x 512 x n, where n is the number of axial scans. There are about 200 images in each CT scan. 

There were a total of 551065 annotations. Of all the annotations provided, 1351 were labeled as nodules, rest were labeled negative. So there big class imbalance. The easy way to deal with it to under sample the majority class and augment the minority class through rotating images. 


We could potentially train the CNN on all the pixels, but that would increase the computational cost and training time. So instead I just decided to crop the images around the coordinates provided in the annotations. The annotation were provided in Cartesian coordinates. So they had to be converted to voxel coordinates. Also the image intensity was defined in Hounsfield scale. So it had to be rescaled for image processing purposes.

The [script](https://github.com/swethasubramanian/LungCancerDetection/blob/master/src/data/create_images.py) below would generate 50 x 50 grayscale images for training, testing and validating a CNN.

<script src="https://gist.github.com/swethasubramanian/8483c5a21d0727e99976b0b9e2b60e68.js"></script>

While the script above under-sampled the negative class such that every 1 in 6 images had a nodule. The data set is still vastly imbalanced for training. I decided to augment my training set by rotating images. The [script](https://github.com/swethasubramanian/LungCancerDetection/blob/master/src/data/augment_images.py) below does just that. 

<script src="https://gist.github.com/swethasubramanian/72697b5cff4c5614c06460885dc7ae23.js"></script>

So for an original image, my script would create these two images:
<table>
    <tr>
      <td><img src="https://cloud.githubusercontent.com/assets/5193925/22851059/03e29daa-efcc-11e6-953f-4d0eba54f7ff.jpg" width="300"/></td>
<td><img src="https://cloud.githubusercontent.com/assets/5193925/22851058/f65f3aee-efcb-11e6-8f7e-e35a3cffb1d9.jpg" width="300"/></td>
<td><img src="https://cloud.githubusercontent.com/assets/5193925/22851043/cccb990c-efcb-11e6-9850-b621d21c8bed.jpg" width="300"/></td>
    </tr>
    <tr><td align="center">original image</td><td align="center">90 degree rotation</td><td align="center">180 degree rotation</td></tr>
</table>

Augmentation resulted in a 80-20 class distribution, which was not entirely ideal. But I also did not want to augment the minority class too much because it might result in a minority class with little variation.

### Building a CNN

Now we are ready to build a CNN. After dabbling a bit with tensorflow, I decided it was way too much work for something incredibly simple. I decided to use [tflearn](http://tflearn.org/). Tflearn is a high-level API wrapper around tensorflow. It made coding lot more palatable. The approach I used was similar to [this](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.if8n708rd). I used a 3 convolutional layers in my architecture.

![arch](https://cloud.githubusercontent.com/assets/5193925/22851323/0b6a1d80-efd3-11e6-9bdf-23ce19796549.png)

My CNN model is defined in a class as shown in the script below.

<script src="https://gist.github.com/swethasubramanian/45be51b64d1595e78fb171c5dbb6cce6.js"></script>

I had a total of 6878 images in my training set. 

### Training the model
Because the data required to train a CNN is very large, it is often desirable to train the model in batches. Loading all the training data into memory is not always possible because you need enough memory to handle it and the features too. I was working out of a 2012 Macbook Pro. So I decided to load all the images into a hdfs dataset using h5py library. You can find the script I used to do that [here.](https://github.com/swethasubramanian/LungCancerDetection/blob/master/src/data/build_hdf5_datasets.py)

Once I had the training data in a hdfs dataset, I trained the model using this script.

<script src="https://gist.github.com/swethasubramanian/dca76567afe1c175e016b2ce299cb7fb.js"></script>

The training took a couple of hours on my laptop. Like any engineer, I wanted to see what goes on under the hood. As the filters are of low resolution (5x5), it would be more useful to visualize features maps generated.

So if I pass through this image through the first convolutional layer (50 x 50 x 32), it generates a feature map that looks like this: 
![conv_layer_0](https://cloud.githubusercontent.com/assets/5193925/22851574/5bb733ba-efdb-11e6-8943-b248f4bcaf58.png)

The max pooling layer following the first layer downsampled the feature map by 2. So when the downsampled feature map is passed into the second convolutional layer of 64 5x5 filters, the resulting feature map is:
![conv_layer_1](https://cloud.githubusercontent.com/assets/5193925/22851575/5cf648d8-efdb-11e6-9363-bb4da8c346fa.png)

The feature map generated by the third convolutional layer containing 64 3x3 filters is:
![conv_layer_2](https://cloud.githubusercontent.com/assets/5193925/22851577/5e79d026-efdb-11e6-8a2f-6860716d582b.png)

### Testing data
I tested my CNN model on 1623 images. I had an validation accuracy of 93 %. My model has a precision of 89.3 % and recall of 71.2 %. The model has a specificity of 98.2 %.

Here is the confusion matrix.

![confusion_matrix](https://cloud.githubusercontent.com/assets/5193925/22851647/9ac21532-efdd-11e6-8618-fd46af1da3a3.png)

I looked deeper into the sort of predictions:
False Negative Predictions:
![preds_fns](https://cloud.githubusercontent.com/assets/5193925/22851625/1d578b90-efdd-11e6-82d7-74d6f8fb69c8.png)
False Positive Predictions:
![preds_fps](https://cloud.githubusercontent.com/assets/5193925/22851626/1f47302c-efdd-11e6-8f17-70128e52875a.png)
True Negative Predictions:
![preds_tns](https://cloud.githubusercontent.com/assets/5193925/22851627/211a12e8-efdd-11e6-921f-62a8e55353c6.png)
True Positive Predictions:
![preds_tps](https://cloud.githubusercontent.com/assets/5193925/22851629/22a94886-efdd-11e6-90bb-a7331ceae520.png)





