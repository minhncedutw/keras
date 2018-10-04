'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 4/30/2018
    Last modified(MM/DD/YYYY HH:MM): 4/30/2018 11:32 AM
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
from ImageAugmentation0119 import pil_image_reader, bgr_to_rgb
from ImageAugmentation0119 import ImageDataGenerator, rescale_value, random_transform, standardize, augment_brightness_contrast, augment_noise
from keras.models import Input, Sequential, Model
from keras.layers import Dense, Conv2D, Convolution2D, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#==============================================================================
# Constant Definitions
#==============================================================================
DATADIR = 'E:/PROJECTS/NTUT/DATA/Cornell/images'
CIFAR_WEIGHT = 'E:/WORKSPACES/python/keras0/MINHNC/output/keras_checkpoint/cifar10/model.acc.199.hdf5'
SIZE = [32, 32, 3]
RESNET_DEPTH = 20
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
                        filepath='./output/keras_checkpoint/model.loss.{epoch:02d}.hdf5', # string, path to save the model file.
                        monitor='val_loss', # quantity to monitor.
                        save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto', # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period=1, # Interval (number of epochs) between checkpoints.
                        verbose=1), # verbosity mode, 0 or 1.
        ModelCheckpoint(
                        # filepath='./output/keras_checkpoint/model.{epoch:02d}.hdf5', # string, path to save the model file.
                        filepath='./output/keras_checkpoint/model.acc.{epoch:02d}.hdf5',
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
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=np.sqrt(0.1),
                          patience=4,
                          verbose=1,
                          epsilon=0.01,
                          mode='min',
                          min_lr=1e-6,
                          cooldown=0),
    ]
    return callbacks_list

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first_layer_but_not_first_stack = stack > 0 and res_block == 0
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_cifar():
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
    X_train = np.array(X_train[:100]).astype('float32')
    y_train = np.array(y_train[:100])
    X_valid = np.array(X_valid[:100]).astype('float32')
    y_valid = np.array(y_valid[:100])
    y_train = to_categorical(y_train, num_classes=10)
    y_valid = to_categorical(y_valid, num_classes=10)

    model = resnet_v1(input_shape=X_train.shape[1:], depth=20)

    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    datagen = ImageDataGenerator()
    datagen.set_pipeline([rescale_value, random_transform, standardize, augment_brightness_contrast])
    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                                  validation_data=datagen.flow(X_valid, y_valid, batch_size=32),
                                  steps_per_epoch=X_train.shape[0]/32*2,
                                  validation_steps=X_valid.shape[0]/32*2,
                                  epochs=1,
                                  callbacks=create_callbacks(),
                                  verbose=1, workers=1)

    accuracy = model.evaluate(x=X_valid, y=y_valid, batch_size=32, verbose=1)
    print(accuracy)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return 0

def load_data(datadir):
    X = []
    y = []
    all_files = glob(os.path.join(datadir, 'graspableRGB/*jpg'))
    for RGBFile in all_files:
        img_RGB = pil_image_reader(RGBFile, target_size=SIZE[:2], keep_ratio=True)
        X.append(img_RGB)
        y.append(1)

    all_files = glob(os.path.join(datadir, 'ungraspableRGB/*jpg'))
    for RGBFile in all_files:
        selected = random.uniform(0.0, 1.0)
        if selected > 0.2:
            continue
        img_RGB = pil_image_reader(RGBFile, target_size=SIZE[:2], keep_ratio=True)
        X.append(img_RGB)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    X = X.astype(np.float32) / 255.0
    y = keras.utils.to_categorical(y, 2)
    return X, y

def train_cornell():
    X, y = load_data(DATADIR)
    X = np.array(X)
    y = np.array(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True)

    cifar_model = resnet_v1(input_shape=SIZE, depth=RESNET_DEPTH)
    cifar_model.load_weights(CIFAR_WEIGHT)

    cornell_model = resnet_v1(input_shape=SIZE, depth=RESNET_DEPTH, num_classes=2)
    for i in range(47):
        cornell_model.layers[i] = cifar_model.layers[i]
        cornell_model.layers[i].trainable = False

    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    cornell_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    train_gen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True)
    train_gen.set_pipeline([random_transform, standardize, augment_brightness_contrast, augment_noise])
    train_gen.fit(X)

    for i in range(3):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True)
        history = cornell_model.fit_generator(train_gen.flow(X_train, y_train, batch_size=32), steps_per_epoch=X_train.shape[0]/32,
                                          validation_data=train_gen.flow(X_valid, y_valid, batch_size=32), validation_steps=X_valid.shape[0]/32,
                                          epochs=10,
                                          callbacks=create_callbacks(),
                                          verbose=1)
        import time
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        time.sleep(5)

    accuracy = cornell_model.evaluate(x=X_valid, y=y_valid, batch_size=32, verbose=1)
    print(accuracy)

    return 0
#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    train_cornell()


if __name__ == '__main__':
    main()
