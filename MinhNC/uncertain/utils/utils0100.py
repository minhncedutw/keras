'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 6/18/2018
    Last modified(MM/DD/YYYY HH:MM): 6/18/2018 3:51 PM
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
import os

print(os.getcwd())
import sys
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import cv2

import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers.merge import concatenate

import imgaug as ia
from imgaug import augmenters as iaa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#==============================================================================
# Constant Definitions
#==============================================================================
PI = 3.14
def DEGREE_2_RADIAN(X):
    return (X * PI / 180)
def RADIAN_2_DEGREE(X):
    return (X*180/PI)
def M_2_MM(X):
    return (X*1000)
def MM_2_M(X):
    return (X/1000.0)

class CMImageType:
    COLOR = cv2.IMREAD_COLOR
    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    UNCHANGED = cv2.IMREAD_UNCHANGED

class CMColor:
    # Opencv color definition
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    SKY = (255, 255, 0)
    PINK = (255, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)

#==============================================================================
# Function Definitions
#==============================================================================
def show_image(image, windowname=None, wait=None):
    if windowname is None:
        windowname = "Unknown"
    cv2.namedWindow(winname=windowname, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname=windowname, mat=image)
    if wait is not None:
        cv2.waitKey(wait)
    return 0

def plot_rectangle(image, rect_points, rc_or_xy='xy', edge1_color=CMColor.RED, edge2_color=CMColor.GREEN, edge3_color=CMColor.YELLOW, edge4_color=CMColor.BLUE, size=1):
    rect_points = np.array(rect_points).astype(np.int32)

    if rc_or_xy=='xy':
        x1, y1, x2, y2, x3, y3, x4, y4 = rect_points[0, 0], rect_points[0, 1], \
                                         rect_points[1, 0], rect_points[1, 1], \
                                         rect_points[2, 0], rect_points[2, 1], \
                                         rect_points[3, 0], rect_points[3, 1]
    else:
        x1, y1, x2, y2, x3, y3, x4, y4 = rect_points[0, 1], rect_points[0, 0], \
                                         rect_points[1, 1], rect_points[1, 0], \
                                         rect_points[2, 1], rect_points[2, 0], \
                                         rect_points[3, 1], rect_points[3, 0]
    _image = np.copy(image)
    cv2.line(_image, (x1, y1), (x2, y2), edge1_color, size)
    cv2.line(_image, (x2, y2), (x3, y3), edge2_color, size)
    cv2.line(_image, (x3, y3), (x4, y4), edge3_color, size)
    cv2.line(_image, (x4, y4), (x1, y1), edge4_color, size)
    return _image

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        for i in range(len(argv) - 1):
            print(argv[i + 1])


if __name__ == '__main__':
    main()
