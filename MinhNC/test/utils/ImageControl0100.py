'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 5/12/2018
    Last modified(MM/DD/YYYY HH:MM): 5/12/2018 6:05 PM
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
import copy
import threading
import inspect
import types

import numpy as np
from scipy import linalg

import keras.backend as K
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis

#==============================================================================
# Constant Definitions
#==============================================================================
import cv2

def bgr_to_rgb(image, dim_ordering='tf'):
    if dim_ordering == 'tf':
        image_new = image[..., ::-1]
    elif dim_ordering == 'th':
        image_new = image[::-1, ...]
    return image_new

def cv_resize_image(image, target_size):
    new_image = cv2.resize(image, (target_size[1], target_size[0]))
    return new_image

def cv_resize_pad_image(image, target_size):
    old_size = image.shape[:2]
    ratio = np.min((float(target_size[0])/old_size[0], float(target_size[1])/old_size[1]))
    new_size = tuple([int(size * ratio) for size in old_size])
    new_image = cv2.resize(image, (new_size[1], new_size[0]))

    # calculate pad sizes
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # pad image by black color
    color = [0, 0, 0] # black color
    new_image = cv2.copyMakeBorder(new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def cv_load_image(path, flag=cv2.IMREAD_COLOR, image_type='RGB', target_mode=None, target_size=None, keep_ratio=False):
    """
        Reference resize-pad: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    :param path:
    :param target_mode:
    :param target_size:
    :param keep_ratio:
    :return:
    """
    image = cv2.imread(filename=path, flags=flag)
    # read image of opencv is BGR image, so it's necessary to convert to RGB image.
    if image_type=='RGB':
        image = bgr_to_rgb(image=image, dim_ordering='tf')

    # resize image
    if target_size is not None:
        # SHOULD do checking whether size is bigger then 0
            #Add checking code here
        #==============================================
        if keep_ratio: # resize and keep ratio(need to pad)
            image = cv_resize_pad_image(image=image, target_size=target_size)
        else: # resize and need not keep ratio(image will be distorted)
            image = cv_resize_image(image=image, target_size=target_size)
    return image

def cv_show_image(image, window_name=None, wait=None):
    if window_name is None:
        window_name = "Unknown"
    cv2.namedWindow(winname=window_name, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname=window_name, mat=image)
    if wait is not None:
        cv2.waitKey(wait)
    return True

def cv_plot_rectangle(image, rect_points, edge1_color, edge2_color, edge3_color, edge4_color, size):
    new_image = np.copy(image) # copy image
    # get (row, column) of point list
    pt1_r = rect_points[0, 0] # point 1 row
    pt1_c = rect_points[0, 1] # point 1 col
    pt2_r = rect_points[1, 0] # ...
    pt2_c = rect_points[1, 1]
    pt3_r = rect_points[2, 0]
    pt3_c = rect_points[2, 1]
    pt4_r = rect_points[3, 0]
    pt4_c = rect_points[3, 1]

    cv2.line(new_image, pt1=(pt1_c, pt1_r), pt2=(pt2_c, pt2_r), color=edge1_color, thickness=size)
    cv2.line(new_image, pt1=(pt2_c, pt2_r), pt2=(pt3_c, pt3_r), color=edge2_color, thickness=size)
    cv2.line(new_image, pt1=(pt3_c, pt3_r), pt2=(pt4_c, pt4_r), color=edge3_color, thickness=size)
    cv2.line(new_image, pt1=(pt4_c, pt4_r), pt2=(pt1_c, pt1_r), color=edge4_color, thickness=size)
    return new_image


from scipy import ndimage
def rotate_image(image, angle_degree, reshape=False):
    new_image = ndimage.rotate(input=image, angle=angle_degree, reshape=reshape)
    return new_image

def rotate_around_dynamic_point(self, originalpoint, firstcenter, lastcenter, angle_degree):
    radangle = angle_degree / 180 * np.pi

    originalpoint = np.array([originalpoint[0], originalpoint[1], 1])
    translatematrix1 = np.array([[1, 0, -firstcenter[0]], [0, 1, -firstcenter[1]], [0, 0, 1]])
    rotatematrix = np.array(
        [[np.cos(radangle), -np.sin(radangle), 0], [np.sin(radangle), np.cos(radangle), 0], [0, 0, 1]])
    translatematrix2 = np.array([[1, 0, lastcenter[0]], [0, 1, lastcenter[1]], [0, 0, 1]])

    rotatedpoint = np.dot(translatematrix2, np.dot(rotatematrix, np.dot(translatematrix1, originalpoint)))
    return np.array([rotatedpoint[0], rotatedpoint[1]])


from PIL import Image, ImageOps
def pil_resize_img(img, target_size=None, keep_ratio=False, **kwargs):
    if keep_ratio == False:
        img_new = img.resize((target_size[1], target_size[0]))
        # img_new = imresize(img, (target_size[1], target_size[0]))
    else:
        old_size = img.size
        ratio = min([i / j for i, j in zip(target_size, old_size[:2])])
        new_size = tuple([int(size * ratio) for size in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        # img = imresize(img, new_size)

        img_new = Image.new("RGB", target_size[:2])
        img_new.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return img_new

def resize_images(images, target_size=None, keep_ratio=False, **kwargs):
    new_images = []
    for image in images:
        new_image = pil_resize_img(img=image, target_size=target_size, keep_ratio=keep_ratio)
        new_images.append(new_image)
    return np.array(new_images)

def pil_load_img(path, target_mode=None, target_size=None, keep_ratio=False):
    """
        Reference resize-pad: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    :param path:
    :param target_mode:
    :param target_size:
    :param keep_ratio:
    :return:
    """
    img = Image.open(path)
    if target_mode:
        img = img.convert(target_mode)
    if target_size:
        img = pil_resize_img(img, target_size=target_size, keep_ratio=keep_ratio)
    return img

def pil_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), keep_ratio=False, **kwargs):
    img = pil_load_img(filepath, target_mode=target_mode, target_size=target_size, keep_ratio=keep_ratio)
    return img_to_array(img, dim_ordering=dim_ordering)

def array_to_img(x, dim_ordering=K.image_dim_ordering(), mode=None, scale=True):
    x = x.copy()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3 and mode == 'RGB':
        return Image.fromarray(x.astype('uint8'), mode)
    elif x.shape[2] == 1 and mode == 'L':
        return Image.fromarray(x[:, :, 0].astype('uint8'), mode)
    elif mode:
        return Image.fromarray(x, mode)
    else:
        raise Exception('Unsupported array shape: ', x.shape)

def img_to_array(img, dim_ordering=K.image_dim_ordering()):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x

def standardize(x,
                dim_ordering='th',
                rescale=False,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                mean=None, std=None,
                samplewise_std_normalization=False,
                zca_whitening=False, principal_components=None,
                featurewise_standardize_axis=None,
                samplewise_standardize_axis=None,
                fitting=False,
                verbose=0,
                config={},
                **kwargs):
    '''
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
    '''
    if fitting:
        if '_X' in config:
            # add data to _X array
            config['_X'][config['_iX']] = x
            config['_iX'] += 1
            # if verbose and config.has_key('_fit_progressbar'):
            #     config['_fit_progressbar'].update(config['_iX'], force=(config['_iX'] == fitting))

            # the array (_X) is ready to fit
            if config['_iX'] >= fitting:
                X = config['_X'].astype('float32')
                del config['_X']
                del config['_iX']
                if featurewise_center or featurewise_std_normalization:
                    featurewise_standardize_axis = featurewise_standardize_axis or 0
                    if type(featurewise_standardize_axis) is int:
                        featurewise_standardize_axis = (featurewise_standardize_axis,)
                    assert 0 in featurewise_standardize_axis, 'feature-wise standardize axis should include 0'

                if featurewise_center:
                    mean = np.mean(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['mean'] = np.squeeze(mean, axis=0)
                    X -= mean

                if featurewise_std_normalization:
                    std = np.std(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['std'] = np.squeeze(std, axis=0)
                    X /= (std + 1e-7)

                if zca_whitening:
                    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
                    sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
                    U, S, V = linalg.svd(sigma)
                    config['principal_components'] = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
                # if verbose:
                #     del config['_fit_progressbar']
        else:
            # start a new fitting, fitting = total sample number
            config['_X'] = np.zeros((fitting,) + x.shape)
            config['_iX'] = 0
            config['_X'][config['_iX']] = x
            config['_iX'] += 1
            # if verbose:
            #     config['_fit_progressbar'] = Progbar(target=fitting, verbose=verbose)
        return x

    if rescale:
        x *= rescale

    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        channel_index = 0
    if dim_ordering == 'tf':
        channel_index = 2

    samplewise_standardize_axis = samplewise_standardize_axis or channel_index
    if type(samplewise_standardize_axis) is int:
        samplewise_standardize_axis = (samplewise_standardize_axis,)

    if samplewise_center:
        x -= np.mean(x, axis=samplewise_standardize_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=samplewise_standardize_axis, keepdims=True) + 1e-7)

    if verbose:
        if (featurewise_center and mean is None) or (featurewise_std_normalization and std is None) or (
            zca_whitening and principal_components is None):
            print('WARNING: feature-wise standardization and zca whitening will be disabled, please run "fit" first.')

    if featurewise_center:
        if mean is not None:
            x -= mean
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)

    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def random_transform(x,
                     dim_ordering='th',
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rescale=None,
                     sync_seed=None,

                     **kwargs):
    '''
    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
    '''
    np.random.seed(sync_seed)

    x = x.astype('float32')
    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        img_channel_index = 0
        img_row_index = 1
        img_col_index = 2
    if dim_ordering == 'tf':
        img_channel_index = 2
        img_row_index = 0
        img_col_index = 1
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, img_channel_index)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)

    # TODO:
    # barrel/fisheye

    np.random.seed()
    return x

class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        seed: random seed for reproducible pipeline processing. If not None, it will also be used by `flow` or
            `flow_from_directory` to generate the shuffle index in case of no seed is set.
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 featurewise_standardize_axis=None,
                 samplewise_standardize_axis=None,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering=K.image_dim_ordering(),
                 seed=None,
                 verbose=1):
        self.config = copy.deepcopy(locals())
        self.config['config'] = self.config
        self.config['mean'] = None
        self.config['std'] = None
        self.config['principal_components'] = None
        self.config['rescale'] = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)

        self.__sync_seed = self.config['seed'] or np.random.randint(0, 4294967295, dtype=np.int64)

        self.default_pipeline = []
        self.default_pipeline.append(random_transform)
        self.default_pipeline.append(standardize)
        self.set_pipeline(self.default_pipeline)

        self.__fitting = False
        self.fit_lock = threading.Lock()

    @property
    def sync_seed(self):
        return self.__sync_seed

    @property
    def fitting(self):
        return self.__fitting

    @property
    def pipeline(self):
        return self.__pipeline

    def sync(self, image_data_generator):
        self.__sync_seed = image_data_generator.sync_seed
        return (self, image_data_generator)

    def set_pipeline(self, p):
        if p is None:
            self.__pipeline = self.default_pipeline
        elif type(p) is list:
            self.__pipeline = p
        else:
            raise Exception('invalid pipeline.')

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_mode=None, save_format='jpeg'):

        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.config['dim_ordering'],
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    def flow_from_directory(self, directory,
                            color_mode=None, target_size=None,
                            image_reader='pil', reader_config=None,
                            read_formats=None,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='',
                            save_mode=None, save_format='jpeg'):
        if reader_config is None:
            reader_config = {'target_mode': 'RGB', 'target_size': (256, 256)}
        if read_formats is None:
            read_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        return DirectoryIterator(
            directory, self,
            color_mode=color_mode, target_size=target_size,
            image_reader=image_reader, reader_config=reader_config,
            read_formats=read_formats,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.config['dim_ordering'],
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    def process(self, x):
        # get next sync_seed
        np.random.seed(self.__sync_seed)

        self.__sync_seed = np.random.randint(0, 4294967295, dtype=np.int64)

        self.config['fitting'] = self.__fitting

        self.config['sync_seed'] = self.__sync_seed
        for p in self.__pipeline:
            x = p(x, **self.config)
        return x

    def fit_generator(self, generator, nb_iter):
        '''Fit a generator
        # Arguments
            generator: Iterator, generate data for fitting.
            nb_iter: Int, number of iteration to fit.
        '''
        with self.fit_lock:
            try:
                self.__fitting = nb_iter * generator.batch_size
                for i in range(nb_iter):
                    next(generator)
            finally:
                self.__fitting = False

    def fit(self, X, rounds=1):
        '''Fit the pipeline on a numpy array
        # Arguments
            X: Numpy array, the data to fit on.
            rounds: how many rounds of fit to do over the data
        '''
        X = np.copy(X)
        with self.fit_lock:
            try:
                self.__fitting = rounds * X.shape[0]
                for r in range(rounds):
                    for i in range(X.shape[0]):
                        self.process(X[i])
            finally:
                self.__fitting = False

class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                self.index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    self.index_array = np.random.permutation(N)
                    if seed is not None:
                        np.random.seed()

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __add__(self, it):
        assert self.N == it.N
        assert self.batch_size == it.batch_size
        assert self.shuffle == it.shuffle
        seed = self.seed or np.random.randint(0, 4294967295, dtype=np.int64)
        it.total_batches_seen = self.total_batches_seen
        self.index_generator = self._flow_index(self.N, self.batch_size, self.shuffle, seed)
        it.index_generator = it._flow_index(it.N, it.batch_size, it.shuffle, seed)
        if (sys.version_info > (3, 0)):
            iter_zip = zip
        else:
            from itertools import izip
            iter_zip = izip
        return iter_zip(self, it)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class NumpyArrayIterator(Iterator):
    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering=K.image_dim_ordering(),
                 save_to_dir=None, save_prefix='',
                 save_mode=None, save_format='jpeg'):

        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (
                            np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_mode = save_mode
        self.save_format = save_format
        seed = seed or image_data_generator.config['seed']

        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def __add__(self, it):
        if isinstance(it, NumpyArrayIterator):
            assert self.X.shape[0] == it.X.shape[0]
        if isinstance(it, DirectoryIterator):
            assert self.X.shape[0] == it.nb_sample
        it.image_data_generator.sync(self.image_data_generator)
        return super(NumpyArrayIterator, self).__add__(it)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_x = None

        for i, j in enumerate(index_array):

            x = self.X[j]
            x = self.image_data_generator.process(x)
            if i == 0:
                batch_x = np.zeros((current_batch_size,) + x.shape)

            batch_x[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, mode=self.save_mode, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

class DirectoryIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 color_mode=None, target_size=None,
                 image_reader="pil", read_formats=None,
                 reader_config=None,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='',
                 save_mode=None, save_format='jpeg'):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.image_reader = image_reader
        if self.image_reader == 'pil':
            self.image_reader = pil_image_reader
        if read_formats is None:
            read_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        if reader_config is None:
            reader_config = {'target_mode': 'RGB', 'target_size': None}
        self.reader_config = reader_config
        # TODO: move color_mode and target_size to reader_config
        if color_mode == 'rgb':
            self.reader_config['target_mode'] = 'RGB'
        elif color_mode == 'grayscale':
            self.reader_config['target_mode'] = 'L'

        if target_size:
            self.reader_config['target_size'] = target_size

        self.dim_ordering = dim_ordering
        self.reader_config['dim_ordering'] = dim_ordering
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_mode = save_mode
        self.save_format = save_format

        seed = seed or image_data_generator.config['seed']

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        # if no class is found, add '' for scanning the root folder
        if class_mode is None and len(classes) == 0:
            classes.append('')
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                is_valid = False
                for extension in read_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                is_valid = False
                for extension in read_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.classes[i] = self.class_indices[subdir]
                    self.filenames.append(os.path.join(subdir, fname))
                    i += 1

        assert len(self.filenames) > 0, 'No valid file is found in the target directory.'
        self.reader_config['class_mode'] = self.class_mode
        self.reader_config['classes'] = self.classes
        self.reader_config['filenames'] = self.filenames
        self.reader_config['directory'] = self.directory
        self.reader_config['nb_sample'] = self.nb_sample
        self.reader_config['seed'] = seed
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        if inspect.isgeneratorfunction(self.image_reader):
            self._reader_generator_mode = True
            self._reader_generator = []
            # set index batch_size to 1
            self.index_generator = self._flow_index(self.N, 1, self.shuffle, seed)
        else:
            self._reader_generator_mode = False

    def __add__(self, it):
        if isinstance(it, DirectoryIterator):
            assert self.nb_sample == it.nb_sample
            assert len(self.filenames) == len(it.filenames)
            assert np.alltrue(self.classes == it.classes)
            assert self.image_reader == it.image_reader
            if inspect.isgeneratorfunction(self.image_reader):
                self._reader_generator = []
                it._reader_generator = []
        if isinstance(it, NumpyArrayIterator):
            assert self.nb_sample == self.X.shape[0]
        it.image_data_generator.sync(self.image_data_generator)
        return super(DirectoryIterator, self).__add__(it)

    def next(self):
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        if self._reader_generator_mode:
            sampleCount = 0
            batch_x = None
            _new_generator_flag = False
            while sampleCount < self.batch_size:
                for x in self._reader_generator:
                    _new_generator_flag = False
                    if x.ndim == 2:
                        x = np.expand_dims(x, axis=0)
                    x = self.image_data_generator.process(x)
                    self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
                    if sampleCount == 0:
                        batch_x = np.zeros((self.batch_size,) + x.shape)
                    batch_x[sampleCount] = x
                    sampleCount += 1
                    if sampleCount >= self.batch_size:
                        break
                if sampleCount >= self.batch_size or _new_generator_flag:
                    break
                with self.lock:
                    index_array, _, _ = next(self.index_generator)
                fname = self.filenames[index_array[0]]
                self._reader_generator = self.image_reader(os.path.join(self.directory, fname),
                                                           **self.reader_config)
                assert isinstance(self._reader_generator, types.GeneratorType)
                _new_generator_flag = True
        else:
            with self.lock:
                index_array, current_index, current_batch_size = next(self.index_generator)
            # The transformation of images is not under thread lock so it can be done in parallel
            batch_x = None
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
                if x.ndim == 2:
                    x = np.expand_dims(x, axis=0)
                x = self.image_data_generator.process(x)
                if i == 0:
                    batch_x = np.zeros((current_batch_size,) + x.shape)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, mode=self.save_mode, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


def rescale_value(img, **kwargs):
    img = img.astype('float32') * 1./255
    return img

def manipulate_brighness_contrast(image, alpha=1, beta=0, **kwargs):
    image_new = image*alpha + beta
    image_new = np.clip(image_new, 0., 1.)
    return image_new

def manipulate_gamma(image, gamma=1, **kwargs):
    image_new = np.power(image, gamma)
    image_new = np.clip(image_new, 0., 1.)
    return image_new

def augment_noise(image, **kwargs):
    """
    Reference: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    :param image:
    :param kwargs:
    :return:
    """
    rate = np.random.random()
    if rate < 0.2: # "gauss"
        row, col, ch = image.shape
        mean = 0
        var = 0.002
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image_noise = image + gauss
        image_noise = np.clip(image_noise, 0., 1.)
    elif rate < 0.4: # "s&p"
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        image_noise = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        image_noise[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image_noise[coords] = 0
    elif rate < 0.6: # "poisson"
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image_noise = np.random.poisson(image * vals) / float(vals)
        image_noise = np.clip(image_noise, 0., 1.)
    elif rate < 0.8: # "speckle"
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        image_noise = image + image * gauss
        image_noise = np.clip(image_noise, 0., 1.)
    else:
        image_noise = image
    return image_noise

from skimage import exposure
def augment_brightness_contrast(image, **kwargs):
    rate = np.random.random()
    if rate < 0.33:
        alpha = 0.8 + np.random.random()/2
        beta = -0.125 + np.random.random()/4
        image_new = manipulate_brighness_contrast(image, alpha=alpha, beta=beta)
    elif rate<0.4:
        gamma = np.random.random() / 2
        operator = int(np.round(np.random.random()))
        if operator:
            image_new = manipulate_gamma(image, alpha=gamma)
        else:
            image_new = manipulate_gamma(image, alpha=1./gamma)
    elif rate < 0.6:
        rate2 = np.random.random()
        if rate2 < 0.33: # do contrast stretching
            p2, p98 = np.percentile(image, (2, 98), interpolation='lower')
            image_new = exposure.rescale_intensity(image, in_range=(p2, p98))
        elif rate2 < 0.66: # do equalize histogram
            image_new = exposure.equalize_hist(image)
        else: # do adaptive equalize histogram
            image_new = exposure.equalize_adapthist(image, clip_limit=0.03)
    # elif rate < 0.8:
    #     image_new = manipulate_noise(image)
    else:
        image_new = image
    return image_new

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is Image Control Program')




if __name__ == '__main__':
    main()
