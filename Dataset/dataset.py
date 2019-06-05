#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from utils.misc_utils import get, get_label_info
import logging
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import random


def one_hot_it(label, label_values):
  """
  Convert a segmentation image label array to one-hot format
  by replacing each pixel value with a vector of length num_classes

  # Arguments
      label: The 2D array segmentation image label
      label_values

  # Returns
      A 2D array with the same width and hieght as the input, but
      with a depth size of num_classes
  """
  semantic_map = []
  for colour in label_values:
    # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
    equality = tf.equal(label, colour)
    class_map = tf.reduce_all(equality, axis=-1)
    semantic_map.append(class_map)
  semantic_map = tf.stack(semantic_map, axis=-1)

  return semantic_map


def _apply_with_random_selector(x, func, num_cases, label):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0], label


def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def _parse_function(image_filename, label_filename, img_mean, class_dict):
    img_contents = tf.read_file(image_filename)
    label_contents = tf.read_file(label_filename)

    # Decode image & label
    img = tf.image.decode_png(img_contents, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    if img_mean is not None:
        img -= img_mean/255

    label = tf.image.decode_png(label_contents, channels=3)
    _, label_values = get_label_info(class_dict)
    label = one_hot_it(label, label_values)
    label = tf.to_int32(label)
    label = tf.argmax(label, axis=-1, output_type=tf.int32)
    label = tf.expand_dims(label, axis=-1)

    return img, label


def _image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)

    return img, label


def _image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=[0])

    return img, label


def _random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w):
    label = tf.to_float(label)
    image_shape = tf.shape(image)
    # TODO: only useful in camvid,  to fix
    pad_h = tf.maximum(crop_h, image_shape[0])-image_shape[0]
    pad_w = tf.maximum(crop_w, image_shape[1])-image_shape[0]
    image = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], constant_values=0)
    label = tf.pad(label, [[0, pad_h], [0, pad_w], [0, 0]], constant_values=30)

    combined = tf.concat(axis=2, values=[image, label])
    # combined = tf.image.pad_to_bounding_box(
    #                         combined,
    #                         0,
    #                         0,
    #                         tf.maximum(crop_h, image_shape[0]),
    #                         tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined, [crop_h, crop_w, last_image_dim+last_label_dim])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = tf.cast(label_crop, dtype=tf.int32)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    # label_crop = tf.image.resize_nearest_neighbor(tf.expand_dims(label_crop, 0), [crop_h//8, crop_w//8])
    # label_crop = tf.squeeze(label_crop, axis=0)

    return img_crop, label_crop


def _check_size(image, label, crop_h, crop_w):
    new_shape = tf.squeeze(tf.stack([[crop_h], [crop_w]]), axis=[1])
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=[0])
    # Set static shape so that tensorflow knows shape at compile time.
    image.set_shape((crop_h, crop_w, 3))
    label.set_shape((crop_h, crop_w, 1))
    label = tf.squeeze(label, axis=2)
    return image, label


class DataLoader(object):
    def __init__(self, config, Dataset='CamVid', class_dict='./CamVid/class_dict.csv'):
        self.config = config
        self.dataSet_dir = Dataset
        self.class_dict = class_dict
        self.dataset = None
        self.iterator = None
        self.build()

    def build(self):
        self.prepare_data()
        self.build_iterator()

    def prepare_data(self):
        # Parameter prepare
        dataset_dir = self.dataSet_dir
        input_dir = self.config['input_dir']
        output_dir = self.config['output_dir']
        crop_h = self.config['crop_h']
        crop_w = self.config['crop_w']
        threads = self.config['prefetch_threads']
        img_mean = get(self.config, 'img_mean', None)
        preprocess_name = get(self.config, 'preprocessing_name', None)
        random_scale = get(self.config, 'random_scale', False)
        random_mirror = get(self.config, 'random_mirror', True)
        batch_size = get(self.config, 'batch_size', 8)

        input_names = []
        output_names = []
        for file in os.listdir(osp.join(dataset_dir, input_dir)):
            cwd = os.getcwd()
            input_names.append(cwd + "/" + osp.join(dataset_dir, input_dir) + "/" + file)
        for file in os.listdir(osp.join(dataset_dir, output_dir)):
            cwd = os.getcwd()
            output_names.append(cwd + "/" + osp.join(dataset_dir, output_dir) + "/" + file)

        input_names.sort(), output_names.sort()

        dataset = tf.data.Dataset.from_tensor_slices((input_names, output_names))
        dataset = dataset.map(lambda x, y: _parse_function(x, y, img_mean, self.class_dict), num_parallel_calls=threads)

        logging.info('preproces -- {}'.format(preprocess_name))
        if preprocess_name == 'augment':
            if random_mirror:
                dataset = dataset.map(_image_mirroring, num_parallel_calls=threads)
            if random_scale:
                dataset = dataset.map(_image_scaling, num_parallel_calls=threads)

            dataset = dataset.map(lambda x, y: _random_crop_and_pad_image_and_labels(x, y, crop_h, crop_w),
                              num_parallel_calls=threads)
            dataset = dataset.map(lambda image, label: _apply_with_random_selector(image, lambda x, ordering: _distort_color
                                                                               (x, ordering, fast_mode=True),
                                                                               num_cases=4, label=label))

        dataset = dataset.map(lambda image, label: _check_size(image, label, crop_h, crop_w))
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        self.dataset = dataset

    def build_iterator(self):
        self.iterator = self.dataset.make_one_shot_iterator()

    def get_one_batch(self):
        return self.iterator.get_next()





