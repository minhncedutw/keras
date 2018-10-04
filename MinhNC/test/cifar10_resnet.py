'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 1/23/2018
    Last modified(MM/DD/YYYY HH:MM): 1/23/2018 6:55 AM
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
from ImageAugmentation0119 import bgr_to_rgb, ImageDataGenerator, rescale_value, random_transform, standardize, augment_brightness_contrast, augment_noise
from keras.models import Input, Sequential, Model
from keras.layers import Dense, Conv2D, Convolution2D, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, Dropout, Flatten, BatchNormalization, concatenate, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#==============================================================================
# Constant Definitions
#==============================================================================
# DATADIR = 'E:/PROJECTS/NTUT/DATA/Cornell/images'
DATADIR = 'E:/PROJECTS/NTUT/DATA/Cornell/images'
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

def resnet_v0(input_shape, num_classes, num_filters, num_blocks, num_sub_blocks, is_using_max_pool):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=7, strides=2, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
               padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if is_using_max_pool:
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for i in range(num_blocks):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = Conv2D(num_filters, kernel_size=3, strides=strides, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4), padding='same')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                       padding='same')(y)
            y = BatchNormalization()(y)
            if is_first_layer_but_not_first_block:
                x = Conv2D(num_filters, kernel_size=1, strides=2, kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4), padding='same')(x)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    x = AveragePooling2D()(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=0.001)
    metrics = ['accuracy']
    return model, optimizer, metrics

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


def load_cifar_dataset():
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

    X_train = np.array(X_train[:]).astype('float32')
    y_train = np.array(y_train[:])
    X_valid = np.array(X_valid[:]).astype('float32')
    y_valid = np.array(y_valid[:])
    y_train = to_categorical(y_train, num_classes=10)
    y_valid = to_categorical(y_valid, num_classes=10)
    return X_train, y_train, X_valid, y_valid

def load_cornell_dataset(datadir):
    X = []
    y = []
    grasable_files = glob(os.path.join(datadir, 'graspableRGB/*jpg'))
    for RGBFile in grasable_files:
        BGRI = cv2.imread(RGBFile, cv2.IMREAD_COLOR)
        RGBI = bgr_to_rgb(BGRI)
        RGBI = cv2.resize(RGBI, (SIZE[0], SIZE[1]))
        X.append(RGBI)
        y.append(1)
        # BGRI = cv2.imread(RGBFile, cv2.IMREAD_COLOR)
        # RGBI = bgr_to_rgb(BGRI)
        # X.append(RGBI)
        # y.append(1)

    grasable_files = glob(os.path.join(datadir, 'ungraspableRGB/*jpg'))
    for RGBFile in grasable_files:
        BGRI = cv2.imread(RGBFile, cv2.IMREAD_COLOR)
        RGBI = bgr_to_rgb(BGRI)
        RGBI = cv2.resize(RGBI, (SIZE[0], SIZE[1]))
        X.append(RGBI)
        y.append(0)
        # BGRI = cv2.imread(RGBFile, cv2.IMREAD_COLOR)
        # RGBI = bgr_to_rgb(BGRI)
        # X.append(RGBI)
        # y.append(0)

    X = np.array(X).astype(np.float32) / 255.
    y = keras.utils.to_categorical(y, 2)
    return X, y

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


def train_cifar_normalnet():
    X_train, y_train, X_valid, y_valid = load_cifar_dataset()

    model = normalnet(input_shape=X_train.shape[1:], num_classes=10)
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
                                  epochs=200,
                                  callbacks=create_callbacks(),
                                  verbose=1, workers=1)

    accuracy = model.evaluate(x=X_valid, y=y_valid, batch_size=32, verbose=1)
    print(accuracy)

    # list all data in history
    print(history.history.keys())
    # plot learning curves for accuracy and loss
    kr_plot_history(history, display_acc=True, display_loss=True)
    return 0

def train_cifar_normalnet_fullforward():
    X_train, y_train, X_valid, y_valid = load_cifar_dataset()

    model = normalnet_fullforward(input_shape=X_train.shape[1:], num_classes=10)
    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    datagen = ImageDataGenerator()
    datagen.set_pipeline([rescale_value, random_transform, standardize, augment_brightness_contrast])
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

def train_cornell_normalnet_fullforward():
    X, y = load_cornell_dataset(DATADIR)
    fold_Xtrain = []
    fold_Xtest = []
    fold_ytrain = []
    fold_ytest = []
    for i in range(5):
        data_length = int(X.shape[0] / 5)
        indices = np.arange(start=i*data_length, stop=(i+1)*data_length)
        X_test = [X[i] for i in range(len(X)) if i in indices]
        X_train = [X[i] for i in range(len(X)) if i not in indices]
        y_test = [y[i] for i in range(len(y)) if i in indices]
        y_train = [y[i] for i in range(len(y)) if i not in indices]
        fold_Xtrain.append(X_train)
        fold_ytrain.append(y_train)
        fold_Xtest.append(X_test)
        fold_ytest.append(y_test)

    cifar_model = normalnet_fullforward(input_shape=(28, 28, 3), num_classes=10)
    cifar_model.load_weights(filepath='output/keras_checkpoint/train01/model.acc.156.hdf5')
    model = normalnet_fullforward(input_shape=(28, 28, 3), num_classes=2)
    for i in range(len(model.layers)):
        if model.layers[i].name == 'dense_1':
            break
        model.layers[i].set_weights(cifar_model.layers[i].weights)
    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    datagen = ImageDataGenerator()
    datagen.set_pipeline([rescale_value, random_transform, standardize, augment_brightness_contrast])
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

def train_cifar_resnet():
    X_train, y_train, X_valid, y_valid = load_cifar_dataset()

    model = resnet_v1(input_shape=X_train.shape[1:], depth=20)

    optimizer = Adam(lr=lr_schedule(0))
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    train_gen = ImageDataGenerator(
                                 # rescale=1.0/255,
                                 # randomly shift images horizontally
                                 # width_shift_range=0.1,
                                 # randomly shift images vertically
                                 # height_shift_range=0.1,
                                 # randomly flip images
                                 horizontal_flip=True,
                                 # randomly flip images
                                 vertical_flip=True
                                 )
    train_gen.set_pipeline([rescale_value, random_transform, standardize, augment_brightness_contrast, augment_noise])
    train_gen.fit(X_train)

    valid_gen = ImageDataGenerator()
    valid_gen.set_pipeline([rescale_value, random_transform, standardize])
    valid_gen.fit(X_train)

    history = model.fit_generator(train_gen.flow(X_train, y_train, batch_size=32),
                                  validation_data=valid_gen.flow(X_valid, y_valid, batch_size=32),
                                  steps_per_epoch=X_train.shape[0]/32,
                                  validation_steps=X_valid.shape[0]/32,
                                  epochs=200,
                                  callbacks=create_callbacks(),
                                  verbose=1, workers=1)
    accuracy = model.evaluate_generator(valid_gen.flow(X_valid, y_valid, batch_size=32), steps=X_valid.shape[0]/32, verbose=1)
    print(accuracy)

    # list all data in history
    print(history.history.keys())
    # plot learning curves for accuracy and loss
    kr_plot_history(history, display_acc=True, display_loss=True)
    return 0

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is Graspableness Classification Program')

    train_cornell_normalnet_fullforward()


if __name__ == '__main__':
    main()
