#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#


"""Miscellaneous Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import json
import logging
import os
import re
import sys
from os import path as osp
import csv
import numpy as np


try:
  import pynvml  # nvidia-ml provides utility for NVIDIA management

  HAS_NVML = True
except:
  HAS_NVML = False


def auto_select_gpu():
  """Select gpu which has largest free memory"""
  if HAS_NVML:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    largest_free_mem = 0
    largest_free_idx = 0
    for i in range(deviceCount):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      if info.free > largest_free_mem:
        largest_free_mem = info.free
        largest_free_idx = i
    pynvml.nvmlShutdown()
    largest_free_mem = largest_free_mem / 1024. / 1024.  # Convert to MB

    idx_to_gpu_id = {}
    for i in range(deviceCount):
      idx_to_gpu_id[i] = '{}'.format(i)

    gpu_id = idx_to_gpu_id[largest_free_idx]
    logging.info('Using largest free memory GPU {} with free memory {}MB'.format(gpu_id, largest_free_mem))
    return gpu_id
  else:
    logging.info('nvidia-ml-py is not installed, automatically select gpu is disabled!')
    return '0'


def get_center(x):
  return (x - 1.) / 2.


def get(config, key, default):
  """Get value in config by key, use default if key is not set

  This little function is useful for dynamical experimental settings.
  For example, we can add a new configuration without worrying compatibility with older versions.
  You can also achieve this by just calling config.get(key, default), but add a warning is even better : )
  """
  val = config.get(key)
  if val is None:
    logging.warning('{} is not explicitly specified, using default value: {}'.format(key, default))
    val = default
  return val


def mkdir_p(path):
  """mimic the behavior of mkdir -p in bash"""
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def tryfloat(s):
  try:
    return float(s)
  except:
    return s


def alphanum_key(s):
  """ Turn a string into a list of string and number chunks.
      "z23a" -> ["z", 23, "a"]
  """
  return [tryfloat(c) for c in re.split('([0-9.]+)', s)]


def sort_nicely(l):
  """Sort the given list in the way that humans expect."""
  return sorted(l, key=alphanum_key)


class Tee(object):
  """Mimic the behavior of tee in bash

  From: http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
  Usage:
    tee=Tee('logfile', 'w')
    print 'abcdefg'
    print 'another line'
    tee.close()
    print 'screen only'
    del tee # should do nothing
  """

  def __init__(self, name, mode):
    self.file = open(name, mode)
    self.stdout = sys.stdout
    sys.stdout = self

  def close(self):
    if self.stdout is not None:
      sys.stdout = self.stdout
      self.stdout = None
    if self.file is not None:
      self.file.close()
      self.file = None

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)

  def flush(self):
    self.file.flush()
    self.stdout.flush()

  def __del__(self):
    self.close()


def save_cfgs(train_dir, model_config, train_config):
  """Save all configurations in JSON format for future reference"""
  with open(osp.join(train_dir, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)
  with open(osp.join(train_dir, 'train_config.json'), 'w') as f:
    json.dump(train_config, f, indent=2)


def load_cfgs(checkpoint):
  if osp.isdir(checkpoint):
    train_dir = checkpoint
  else:
    train_dir = osp.dirname(checkpoint)

  with open(osp.join(train_dir, 'model_config.json'), 'r') as f:
    model_config = json.load(f)
  with open(osp.join(train_dir, 'train_config.json'), 'r') as f:
    train_config = json.load(f)

  return model_config, train_config


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
      return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
      file_reader = csv.reader(csvfile, delimiter=',')
      header = next(file_reader)
      for row in file_reader:
        class_names.append(row[0])
        label_values.append([int(row[1]), int(row[2]), int(row[3])])
      # print(class_dict)
    return class_names, label_values

