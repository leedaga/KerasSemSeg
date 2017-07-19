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
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

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

tf_pos_tanh_offset = tf.constant(0.5)
tf_pos_tanh_scale = tf.constant(0.45)

def tanh_zero_to_one(x):
    """Actually [0.05, 0.95] to avoid divide by zero errors"""
    return (tf.tanh(x) * tf_pos_tanh_scale) + tf_pos_tanh_offset

def create_model(opt):
    """Create neural network model, defining layer architecture."""
    model = Sequential()
    
    # Convolution2D(output_depth, convolution height, convolution_width, ...)
    #5x5 trains in 1.5 times duration of 3x3
    #double layer count is linear increase in training time. about 2x
    c = 3
    act = 'tanh'

    model.add(Convolution2D(20, c, c, border_mode='same',
            input_shape=(int((opt['crop_max_y'] - opt['crop_min_y']) / opt['scale_factor']),
                            int((opt['crop_max_x'] - opt['crop_min_x']) / opt['scale_factor']),
                            3)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, c, c, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, c, c, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, c, c, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))    
    model.add(Convolution2D(64, c, c, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, c, c, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(1, c, c, border_mode='same', W_regularizer=l2(0.01), activation=tanh_zero_to_one))
    compile_model(model, opt)

    return model