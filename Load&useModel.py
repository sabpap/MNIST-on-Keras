#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:33:49 2017

1.Load trained model for character recognition
2.Import new character images(created on paint) to test models results


@author: sabpap
"""
#import libraries
import keras.utils 
from keras.models import model_from_json
import numpy as np
from scipy.misc import imread, imresize
from keras.datasets import mnist
from TestDataCreation import load_images_from_folder # my script for new test set creation
from imshow import imshow #custom made imshow function
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

(x_train, y_train), (x_test, y_test) = mnist.load_data() #loading MNIst data for evaluation


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
print("evaluating model...")
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print('best CNN model acc: 99.79%\n')

#testing/playing with new images

#image 1
one = imread('one.png',mode='L')
one = np.invert(one) #inverting image
one = imresize(one,(img_rows, img_cols)) #make it the right size
#imshow(one)
one = one.reshape(1,img_rows, img_cols,1) #convert to a 4D tensor to feed into our model

x = loaded_model.predict_proba(one) #Predict (probability)
x = x.astype(int) #Convert to decision

#image2
four = imread('two.png',mode='L')
four = np.invert(four) #inverting image
four = imresize(four,(img_rows, img_cols)) #make it the right size
#imshow(four)
four = four.reshape(1,img_rows, img_cols,1) #convert to a 4D tensor to feed into our model

y = loaded_model.predict_proba(four) #Predict (probability)
y = y.astype(int) #Convert to decision


#array of images
folder = "NewTestSet"

NewTest = load_images_from_folder(folder,(img_rows,img_cols)) #create new cystom test set

z = loaded_model.predict_proba(NewTest) #Predict (probability)
z = z.astype(int) #Convert to decisions

#print predictions
print('predictions for imported images:') 
for row in z:
    score = np.where(row == 1)
    score = list(score)
    print(int(score[0]))