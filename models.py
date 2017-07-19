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
import tensorflow as tf

def weighted_crossentropy(weight):
    """Higher weights increase the importance of examples in which
    the correct answer is 1. Higher values should be used when
    1 is a rare answer. Lower values should be used when 0 is
    a rare answer."""
    return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def make_model(input_shape, num_classes):
    h, w, ch = input_shape
    model = Sequential()
    model.ch_order = 'channel_last'
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
            output_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_classes, 1, 1, border_mode="same"))
    #model.add(Activation('softmax'))

    presence_weight = 50.0



    model.compile(optimizer='adam',
                loss=weighted_crossentropy(presence_weight),
    #            loss='categorical_crossentropy',
                metrics=['categorical_crossentropy', 'accuracy'])
    #            loss='sparse_categorical_crossentropy',
    #            metrics=['accuracy'])

    return model