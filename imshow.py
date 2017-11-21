#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:00 2017

Custom imshow function for easier use

@author: sabpap
import cv2

"""
from PIL import Image


def imshow(img):
    
    img = Image.fromarray(img)
    img.show()


    return