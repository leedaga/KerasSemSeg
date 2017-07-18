'''
Models
Define the different NN models we will use
Author: Tawn Kramer
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Convolution2D, Reshape
from keras.layers import Dense, Lambda, ELU
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Deconvolution2D

def make_model(input_shape, num_classes, batch_size):
    h, w, ch = input_shape
    model = Sequential()
    model.ch_order = 'channel_last'
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
            output_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(4096, 1, 1, subsample=(1, 1), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_classes, 1, 1, subsample=(1, 1), border_mode="same"))

    #this doesn't work! how to fix?
    #can't seem to get the output size correct
    model.add(Deconvolution2D(num_classes, 64, 64, output_shape=(batch_size, num_classes, h, w), subsample=(32, 32)))
    model.add(Reshape(num_classes,h,w))
    

    model.compile(optimizer="adam", loss="mse")
    return model