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
from keras.layers import Dense, Lambda, ELU, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf
from loss_functions import *

from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.resnet_helpers import conv_block, identity_block, atrous_conv_block, atrous_identity_block

def compile_model(model,opt):
    """Would be part of create_model, except that same settings
        also need to be applied when loading model from file."""
    model.compile(optimizer='adam',
                #loss=weighted_binary_crossentropy(opt['presence_weight']),
                loss=binary_crossentropy_with_logits,
                metrics=['binary_accuracy', 'binary_crossentropy'])

def FCNN(opt):
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
    '''
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
    '''
    
    model.add(Conv2D(num_classes, (1, 1), padding='same', activation='softmax'))
    compile_model(model, opt)

    return model

def AtrousFCN_Resnet50_16s(opt):
    img_input = Input(shape=opt['input_shape'])
    batch_momentum=0.9
    weight_decay=0.
    image_size = opt['input_shape']
    classes = opt['nb_classes']

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=image_size)(x)

    model = Model(img_input, x)
    
    compile_model(model, opt)

    return model

def create_model(opt):
    return AtrousFCN_Resnet50_16s(opt)
    #return FCNN(opt)
