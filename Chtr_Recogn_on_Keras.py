#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:08:00 2017

*Goal :Build CNN for handwritten digits recognition o Keras
*Dataset : MnIST

-Import Libraries
-Load Dataset
-Preprocess Data
-Build Model
-Compile Model
-Train Model 
-Save Model and Weights



@author: sabpap
"""

#libraries

from __future__ import print_function #simple interface for building models
from keras.datasets import mnist
from keras.models import Sequential #type of model:stack of layers
from keras.layers import Dense, Dropout, Flatten #needed types of ANN layers
from keras.layers import Conv2D, MaxPool2D #need types of CNN layers
import keras.utils 

from keras import backend as K

batch_size = 128 # for mini-batch gds
num_classes = 10 #10 different characters
epochs = 30

img_rows, img_cols = 28,28 #input images dimensions

(x_train, y_train), (x_test, y_test) = mnist.load_data() #loading data

#data preprocessing

#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') #convert to float for division
x_test = x_test.astype('float32') #convert to float for division
x_train /= 255  #normalize
x_test /= 255   #normalize
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices(one hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#build model(Architecture details step by step)

model = Sequential() #stack of layers
model.add(Conv2D(32,kernel_size= (3,3), activation= 'relu', input_shape= input_shape)) #conv layer, 32fliters 3x3, RELU
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Dropout(0.25)) # avoid overfitting
model.add(Conv2D(64,kernel_size= (3,3), activation= 'relu')) #conv layer, 64fliters 3x3, RELU
model.add(Flatten()) #matrix to vector
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.25)) # avoid overfitting
model.add(Dense(num_classes, activation= 'softmax'))

model.compile(loss= keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])

#training model 
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

#Save the model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
