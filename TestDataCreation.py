#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:35:30 2017

Image Dataset Creatio(MNIST style)n for testing digit recognition model 

@author: sabpap
"""

import os
import numpy as np
from imshow import imshow
from scipy.misc import imread, imresize

def load_images_from_folder(folder,(img_rows,img_cols)):
    
    num_images = len(os.listdir(folder)) #number of images in folder
    images = np.empty((num_images,img_rows,img_cols,1)) #array to store images
    list_files = sorted(os.listdir(folder)) #list of files in folder sorted alphabetically
    
    for index,filename in enumerate(list_files):
        
        print('importing file %s...' % (filename)) 
        img = imread(os.path.join(folder,filename),mode='L') #importing image
        if img is not None:
            
            img = np.invert(img) #invert image
            img = imresize(img,(img_rows, img_cols)) #make it the right size
            #rescale image (mostly for better display)
            a = 255/(img.max()-img.min())
            b =  -a*img.min()
            img = a*img + b
            img = img.reshape(img_rows, img_cols,1) 
            images[index] = img #store as a tensor
           
    
            
    images = images.astype('float32') 
    return images
