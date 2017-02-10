#!/usr/bin/env python

import sys

from joblib import Parallel, delayed
import multiprocessing

import pickle

import numpy as np
import pandas as pd

import os
import glob

from PIL import Image

import SimpleITK as sitk


class CTScan(object):
	"""
	A class that allows you to read .mhd header data, crop images and 
	generate and save cropped images

    Args:
    filename: .mhd filename
    coords: a numpy array
	"""
	
    def __init__(self, filename = None, coords = None, path = None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self.path = path

    def reset_coords(self, coords):
        """
        updates to new coordinates
        """
        self.coords = coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        path = glob.glob(self.path + self.filename + '.mhd')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_resolution(self):
        """
        Gets resolution of CT images in mm
        """
        return self.ds.GetSpacing()

    def get_origin(self):
        """
        Gets the origin of defined coordinates
        """
        return self.ds.GetOrigin()

    def get_ds(self):
        """
        Returns the data strcuture containing resolution and coordinate info
        """
        return self.ds

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.get_origin()
        resolution = self.get_resolution()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image
    
    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensiona
        """
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y-width/2):int(y+width/2),\
         int(x-width/2):int(x+width/2)]
        return subImage   
    
    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)


def create_data(idx, outDir, width = 50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    scan = CTScan(np.asarray(X_data.loc[idx])[0], \
        np.asarray(X_data.loc[idx])[1:], raw_image_path)
    outfile = outDir  +  str(idx)+ '.jpg'
    scan.save_image(outfile, width)


def main():
    if len(sys.argv) < 2:
        raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
        mode = sys.argv[1]
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Argument not recognized. Has to be train, test or val')

    inpfile = mode + 'data'
    outDir = mode + '/image_'
    X_data = pd.read_pickle(inpfile)
    raw_image_path = '../../data/raw/*/'

    # Parallelizes inorder to generate more than one image at a time
    Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir) for idx in X_data.index)

if __name__ == "__main__":
    main()




        