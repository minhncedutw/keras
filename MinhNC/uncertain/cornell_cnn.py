'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 5/7/2018
    Last modified(MM/DD/YYYY HH:MM): 5/7/2018 12:22 PM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from glob import glob
import random

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
from ImageAugmentation0119 import ImageDataGenerator, rescale_value, random_transform, standardize, augment_brightness_contrast, augment_noise
from keras.models import Input, Sequential, Model
from keras.layers import Dense, Conv2D, Convolution2D, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, BatchNormalization, concatenate, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#==============================================================================
# Constant Definitions
#==============================================================================
DATADIR = 'E:/PROJECTS/NTUT/DATA/ARLAB/images04'
SIZE = [32, 32, 3]
#==============================================================================
# Function Definitions
#==============================================================================
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def create_callbacks():
    callbacks_list = [
        ModelCheckpoint(
                        # filepath='./output/keras_checkpoint/model.{epoch:02d}.hdf5', # string, path to save the model file.
                        filepath='/home/minhnc-lab/WORKSPACES/Python/keras/MINHNC/output/keras_checkpoint/model.loss.{epoch:02d}.hdf5', # string, path to save the model file.
                        monitor='val_loss', # quantity to monitor.
                        save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto', # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period=1, # Interval (number of epochs) between checkpoints.
                        verbose=1), # verbosity mode, 0 or 1.
        ModelCheckpoint(
                        # filepath='./output/keras_checkpoint/model.{epoch:02d}.hdf5', # string, path to save the model file.
                        filepath='/home/minhnc-lab/WORKSPACES/Python/keras/MINHNC/output/keras_checkpoint/model.acc.{epoch:02d}.hdf5',
                        # string, path to save the model file.
                        monitor='val_acc',  # quantity to monitor.
                        save_best_only=True,
                        # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto',
                        # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false',
                        # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period=1,  # Interval (number of epochs) between checkpoints.
                        verbose=1),  # verbosity mode, 0 or 1.
        ReduceLROnPlateau(monitor='val_loss',
                          factor=np.sqrt(0.1),
                          patience=4,
                          verbose=1,
                          epsilon=0.01,
                          mode='min',
                          min_lr=1e-6,
                          cooldown=0),
		LearningRateScheduler(lr_schedule)
    ]
    return callbacks_list

def kr_plot_history(history, display_acc=False, display_loss=False):
    if display_acc: # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    if display_loss: # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    return 0

def kr_get_layer_index(model, layer_name):
    for i in range(len(model.layers)):
        if model.layers[i].name == layer_name:
            return i
    return -1

def kr_load_model(create_model_func, weight_file=None, layer_name=None):
    model = create_model_func
    if weight_file is not None:
        model.load_weights(weight_file)
        if layer_name is not None:
            layer_idx = kr_get_layer_index(model=model, layer_name=layer_name)
            model2 = create_model_func
            model2.layers[layer_idx+1].input = model.layers[layer_idx].output
            model = Model(inputs=model.layers[0].input, outputs=model2.layers[-1].output)
    return model

def kr_set_trainable(model, trainable=False, layer_idxs=None, layer_names=None, end_layer_idx=None, end_layer_name=None):
    if layer_idxs is not None:
        for layer_idx in layer_idxs:
            model.layers[layer_idx] = trainable
    if layer_names is not None:
        for layer_name in layer_names:
            layer_idx = kr_get_layer_index(model=model, layer_name=layer_name)
            model.layers[layer_idx] = trainable
    if end_layer_idx is not None:
        for i in range(end_layer_idx):
            model.layers[i].trainable = trainable
    if end_layer_name is not None:
        for layer in range(model.layers):
            layer.trainable = trainable
            if layer.name == end_layer_name:
                break
    return model

def normalnet(input_shape, num_classes):
    input = Input(shape=input_shape[-3:])

    conv11 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input)
    conv12 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)
    drop1 = Dropout(rate=0.25)(pool1)

    conv21 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    conv22 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    drop2 = Dropout(rate=0.25)(pool2)
    flat2 = Flatten()(drop2)

    dense3 = Dense(units=512, activation='relu')(flat2)
    drop3 = Dropout(rate=0.25)(dense3)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(drop3)

    model = Model(inputs=input, outputs=output)
    return model

def load_cifar_dataset():
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
    X_train = np.array(X_train[:]).astype('float32') / 255.
    y_train = np.array(y_train[:])
    X_valid = np.array(X_valid[:]).astype('float32') / 255.
    y_valid = np.array(y_valid[:])
    y_train = to_categorical(y_train, num_classes=10)
    y_valid = to_categorical(y_valid, num_classes=10)
    return X_train, y_train, X_valid, y_valid

def normalnet_fullforward(input_shape, num_classes):
    input = Input(shape=input_shape[-3:])

    conv11 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input)
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv11)
    conv12 = BatchNormalization()(conv12)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)
    drop1 = Dropout(rate=0.25)(pool1)
    # flat1 = Flatten()(drop1)

    conv21 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    conv21 = BatchNormalization()(conv21)
    conv22 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv21)
    conv22 = BatchNormalization()(conv22)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    drop2 = Dropout(rate=0.25)(pool2)
    flat2 = Flatten()(drop2)

    conv31 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop2)
    conv31 = BatchNormalization()(conv31)
    conv32 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv31)
    conv32 = BatchNormalization()(conv32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv32)
    drop3 = Dropout(rate=0.25)(pool3)
    flat3 = Flatten()(drop3)

    merge = concatenate([flat2, flat3])
    dense3 = Dense(units=512, activation='relu')(merge)
    drop3 = Dropout(rate=0.25)(dense3)

    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(drop3)

    model = Model(inputs=input, outputs=output)
    return model

def train_cifar_normalnet_fullforward():
    X_train, y_train, X_valid, y_valid = load_cifar_dataset()

    model = normalnet_fullforward(input_shape=X_train.shape[1:], num_classes=10)
    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    datagen = ImageDataGenerator()
    datagen.set_pipeline([random_transform, standardize, augment_brightness_contrast])
    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                                  validation_data=datagen.flow(X_valid, y_valid, batch_size=32),
                                  steps_per_epoch=X_train.shape[0]/32,
                                  validation_steps=X_valid.shape[0]/32,
                                  epochs=200,
                                  callbacks=create_callbacks(),
                                  verbose=1, workers=1)

    accuracy = model.evaluate(x=X_valid, y=y_valid, batch_size=32, verbose=1)
    print(accuracy)

    # plot learning curves for accuracy and loss
    kr_plot_history(history, display_acc=True, display_loss=True)
    return 0


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')
    train_cifar_normalnet_fullforward()


if __name__ == '__main__':
    main()
