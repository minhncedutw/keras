'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 6/18/2018
    Last modified(MM/DD/YYYY HH:MM): 6/18/2018 3:56 PM
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
_CORNELL_DATA_DIR = 'E:/PROJECTS/NTUT/DATA/Cornell'

#==============================================================================
# Function Definitions
#==============================================================================
def load_cornell_bboxes(filename):
    '''Create a list with the coordinates of the grasping rectangles. Every
    element is either x or y of a vertex.'''
    with open(filename, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

def parse_cornell_txt_annotaion(data_dir):
    import glob
    count = 0
    train_imgs = []
    test_imgs = []

    folders = range(1, 11)
    folders = ['0' + str(i) if i < 10 else '10' for i in folders]
    for folder_name in folders:
        for filename in glob.glob(os.path.join(data_dir, folder_name, 'pcd' + folder_name + '*r.png')):
            img = {'object': []}
            img['filename'] = filename
            img['height'] = int(480)
            img['width'] = int(640)


            bboxes_file = filename[:-5] + 'cpos.txt'
            bboxes = load_cornell_bboxes(bboxes_file)

            for idx in range(int(len(bboxes)/8)):
                if np.isnan(np.sum(bboxes[idx*8:idx*8+8])):
                    continue
                obj = {}
                obj['name'] = 'graspable'
                obj['x1'] = int(round(float(bboxes[idx*8 + 0])))
                obj['x2'] = int(round(float(bboxes[idx*8 + 1])))
                obj['x3'] = int(round(float(bboxes[idx*8 + 2])))
                obj['x4'] = int(round(float(bboxes[idx*8 + 3])))
                obj['x5'] = int(round(float(bboxes[idx*8 + 4])))
                obj['x6'] = int(round(float(bboxes[idx*8 + 5])))
                obj['x7'] = int(round(float(bboxes[idx*8 + 6])))
                obj['x8'] = int(round(float(bboxes[idx*8 + 7])))

                img['object'] += [obj]

            if len(img['object']) > 0:
                if count % 5 == 0:
                    test_imgs += [img]
                else:
                    train_imgs += [img]
            count += 1

    return train_imgs, test_imgs

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
