#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:33:49 2017

1.Load trained model for character recognition
2.Import new character images(created on paint) to test models resutlt


@author: sabpap
"""

import keras.utils 
from keras.models import model_from_json
import numpy as np
from scipy.misc import imread, imresize



img_rows, img_cols = 28,28 #input images dimensions


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss= keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])



#testing/playing with new images

one = imread('one.png',mode='L')
#imshow(one)
#one.astype('float32')
#one /= 255
one = np.invert(one)
#make it the right size
one = imresize(one,(img_rows, img_cols))
#convert to a 4D tensor to feed into our model
one = one.reshape(1,img_rows, img_cols,1)

x = loaded_model.predict_proba(one)
x = x.astype(int)


four = imread('two.png',mode='L')
#imshow(one)
#one.astype('float32')
#one /= 255
four = np.invert(four)
#make it the right size
four = imresize(four,(img_rows, img_cols))
#convert to a 4D tensor to feed into our model
four = four.reshape(1,img_rows, img_cols,1)

y = loaded_model.predict_proba(four)
y = y.astype(int)
