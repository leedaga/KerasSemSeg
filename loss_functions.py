from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf


# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)


'''
From Eric Lavigne, Udacity
https://github.com/ericlavigne/CarND-Detect-Lane-Lines-And-Vehicles
'''

def weighted_crossentropy(weight):
    """Higher weights increase the importance of examples in which
    the correct answer is 1. Higher values should be used when
    1 is a rare answer. Lower values should be used when 0 is
    a rare answer."""
    return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))

def weighted_binary_crossentropy(weight):
    """Higher weights increase the importance of examples in which
    the correct answer is 1. Higher values should be used when
    1 is a rare answer. Lower values should be used when 0 is
    a rare answer."""
    return (lambda y_true, y_pred: tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight))
