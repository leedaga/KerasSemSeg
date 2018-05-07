'''
Models
Define the different NN models we will use
Author: Tawn Kramer
'''
from __future__ import print_function
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Reshape, Add, UpSampling2D, Multiply, Concatenate
from keras.layers import Dense, Lambda, ELU
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import tensorflow as tf

def weighted_crossentropy(weight):
    """Higher weights increase the importance of examples in which
    the correct answer is 1. Higher values should be used when
    1 is a rare answer. Lower values should be used when 0 is
    a rare answer."""
    return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

'''
From Eric Lavigne, Udacity
https://github.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles
'''

def weighted_binary_crossentropy(weight):
    """Higher weights increase the importance of examples in which
    the correct answer is 1. Higher values should be used when
    1 is a rare answer. Lower values should be used when 0 is
    a rare answer."""
    return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def compile_model(model,opt):
    """Would be part of create_model, except that same settings
        also need to be applied when loading model from file."""
    model.compile(optimizer='adam',
                loss=weighted_binary_crossentropy(opt['presence_weight']),
                metrics=['binary_accuracy', 'binary_crossentropy'])

def create_model(opt):
    """Create neural network model, defining layer architecture."""
    model = Sequential()
    
    # Conv2D(output_depth, convolution height, convolution_width, ...)
    #5x5 trains in 1.5 times duration of 3x3
    #double layer count is linear increase in training time. about 2x
    c = 5
    act = 'relu'
    num_conv = 32

    num_classes = opt['nb_classes']

    model.add(Conv2D(20, (c, c), padding='same', input_shape=opt['input_shape']))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))

    model.add(Conv2D(num_conv, (c, c), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_conv, (c, c), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_conv, (c, c), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))    
    model.add(Conv2D(num_conv, (c, c), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_conv, (c, c), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(num_classes, (1, 1), padding='same', activation='softmax'))
    compile_model(model, opt)

    return model