'''
    File: train.py
    Author : Tawn Kramer
    Date : July 2017
'''
import os
import sys
import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.loss_function import *
import glob
import densenet_fc

def prepare_mask(img, class_colors, thresh):
    '''
    Take an RGB image where different classes of pixels all share a color
    from the list of class_colors. Then create a matrix with a mask for each
    class using the threshold value passed in. Returns the numpy array of masks.
    Note*
    It only seemed to match the densenet_fc output shape when it was transposed.
    '''
    mask_channels = []
    for col in class_colors:
        lower = np.array(col) - thresh
        upper = np.array(col) + thresh
        mask = cv2.inRange(img, lower, upper)
        mask_channels.append(mask)

    return np.array(mask_channels).transpose()


def generator(samples, class_colors, batch_size=32, perc_to_augment=0.5):
    '''
    Rather than keep all data in memory, we will make a function that keeps
    it's state and returns just the latest batch required via the yield command.
    
    We flip each image horizontally and supply it as a another sample.
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images_X = []
            images_Y = []

            for batch_sample in batch_samples:
                image_a_path, image_b_path = batch_sample

                image_a = cv2.imread(image_a_path)
                if image_a is None:
                    continue

                image_b = cv2.imread(image_b_path)
                if image_b is None:
                    continue

                thresh = 20

                image_b = prepare_mask(image_b, class_colors, thresh)

                #image_a = np.array(image_a)
                #image_b = np.array(image_b)

                images_X.append(image_a)
                images_Y.append(image_b)

                #flip image and steering.
                #image_a = np.fliplr(image_a)
                #image_b = np.fliplr(image_b)

                #images_X.append(image_a)
                #images_Y.append(image_b)


            # final np array to submit to training
            X_train = np.array(images_X)
            y_train = np.array(images_Y)
            yield X_train, y_train


def show_model_summary(model):
    '''
    show the model layer details
    '''
    model.summary()
    print("num layers:", len(model.layers))
    #for layer in model.layers:
    #    print(layer.output_shape)


def make_model(nb_classes, input_shape, batch_size):

    model = densenet_fc.DenseNetFCN(input_shape=input_shape, 
            nb_dense_block=5,
            batchsize=batch_size,
            growth_rate=16,
            nb_layers_per_block=4,
            #upsampling_type='upsampling',
            upsampling_type='deconv',
            classes=nb_classes)

    model.compile(loss = binary_crossentropy_with_logits,
                optimizer="sgd")

    return model

def get_filenames(path_mask, seg_search, seg_rep):
    
    files = glob.glob(path_mask)
    train_files = []

    for f in files:
        train_files.append((f ,f.replace(seg_search, seg_rep)))

    return train_files


def make_generators(path_mask, class_colors, batch_size=32):
    '''
    load the job spec from the csv and create some generator for training
    '''
    
    #get the image/steering pairs from the csv files
    lines = get_filenames(path_mask, "_a.", "_b.")

    print("found %d file pairs." % (len(lines)))
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, class_colors, batch_size=batch_size, perc_to_augment=0.0)
    validation_generator = generator(validation_samples, class_colors, batch_size=batch_size, perc_to_augment=0.0)
    
    #double each because we will flip image in generator
    n_train = len(train_samples) * 2
    n_val = len(validation_samples) * 2
    
    return train_generator, validation_generator, n_train, n_val



def train():
    '''
    Use Keras to train an artificial neural network to use end-to-end behavorial cloning to drive a vehicle.
    '''
    path_mask = './data/*_a.png'
    epochs = 10
    batch_size = 2

    class_colors = [
        ([240, 20, 20]), #lane lines
        ([0, 78, 255]), #road
        ([45, 99, 36]), #ground
        ([250, 250, 250]), #sky
    ]

    nb_classes = len(class_colors)

    input_shape = (224, 224, 3)

    train_generator, validation_generator, n_train, n_val = make_generators(path_mask, class_colors=class_colors, batch_size=batch_size)

    model = make_model(nb_classes, input_shape, batch_size)

    show_model_summary(model)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, verbose=0),
        ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(train_generator, 
        samples_per_epoch = n_train,
        validation_data = validation_generator,
        nb_val_samples = n_val,
        nb_epoch=epochs,
        verbose=1,
        callbacks=callbacks)

    print('training complete.')


def predict():
    '''
    take and image and use a trained model to segment it
    '''
    model = load_model("model.h5", 
        custom_objects={'binary_crossentropy_with_logits': binary_crossentropy_with_logits})

    model.compile(loss = binary_crossentropy_with_logits,
                optimizer="sgd")

    image_path = "./data/image_00000000_a.png"

    print("reading image", image_path)
    img = cv2.imread(image_path)

    print("doing forward pass in image segmentation")
    pred = model.predict(img[None, :, :, :])

    print("pred", pred.shape)

    mask = pred[0][:, :, :3]

    print("mask", mask.shape)

    print("writing test.png output")
    cv2.imwrite("test.png", mask)

if __name__ == "__main__":
    do_pred = False
    
    for arg in sys.argv:
        if arg.find('predict') != -1:
            do_pred = True

    if do_pred:
        predict()
    else:
        train()
