#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:33:49 2017

1.Load trained model for character recognition
2.Import new character images(created on paint) to test models resutlt


@author: sabpap
"""
#import libraries
import keras.utils 
from keras.models import model_from_json
import numpy as np
from scipy.misc import imread, imresize
from keras.datasets import mnist

from keras import backend as K


# load model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# compile model
loaded_model.compile(loss= keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])

# load data for evaluation
img_rows, img_cols = 28,28 #input images dimensions
num_classes = 10 #10 different characters

(x_train, y_train), (x_test, y_test) = mnist.load_data() #loading data


#data preprocessing

#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32') #convert to float for division
x_test /= 255   #normalize

print("%d test samples\n" % x_test.shape[0])
y_test = keras.utils.to_categorical(y_test, num_classes)

#evaluate on test data
print("evaluating...")
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%\n" % (loaded_model.metrics_names[1], score[1]*100))

#testing/playing with new images

#image 1
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

#image2
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
