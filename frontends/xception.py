#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

'''
==================================================================
Based on the Xception Paper (https://arxiv.org/pdf/1610.02357.pdf)
==================================================================
'''


def block(input, mid_out_channels, has_proj, stride, dilation=1, expansion=4):
    if has_proj:
        shortcut = slim.separable_conv2d(input, mid_out_channels*expansion, [3, 3], stride=stride)
        shortcut = slim.batch_norm(shortcut)
    else:
        shortcut = input
    residual = slim.separable_conv2d(input, mid_out_channels, [3, 3], stride=stride, rate=dilation)
    residual = slim.batch_norm(residual)
    residual = tf.nn.relu(residual)
    residual = slim.separable_conv2d(residual, mid_out_channels, [3, 3])
    residual = slim.batch_norm(residual)
    residual = tf.nn.relu(residual)
    residual = slim.separable_conv2d(residual, mid_out_channels*expansion, [3, 3])
    residual = slim.batch_norm(residual)
    output = tf.nn.relu(residual+shortcut)
    return output


@slim.add_arg_scope
def make_layers(input, layers, channel, stride, scope, outputs_collections=None):
    with tf.variable_scope(scope, 'stage', [input]) as sc:
        has_proj = True if stride > 1 else False
        with tf.variable_scope('block1'):
            net = block(input, channel, has_proj, stride)
        for i in range(1, layers):
            with tf.variable_scope('block'+str(i+1)):
                net = block(net, channel, False, stride=1)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, net)


def xception(inputs, layers, channels,
             is_training=True,
             reuse=False,
             scope='Xception'):
    '''
    The Xception Model!

    Note:
    The padding is included by default in slim.conv2d to preserve spatial dimensions.
    INPUTS:
    - inputs(Tensor): a 4D Tensor input of shape [batch_size, height, width, num_channels]
    - num_classes(int): the number of classes to predict
    - is_training(bool): Whether or not to train
    OUTPUTS:
    - logits (Tensor): raw, unactivated outputs of the final layer
    - end_points(dict): dictionary containing the outputs for each layer, including the 'Predictions'
                        containing the probabilities of each output.
    '''
    with tf.variable_scope(scope, 'xception', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([make_layers], outputs_collections=end_points_collection), \
             slim.arg_scope([slim.batch_norm], is_training=is_training):
            # ===========ENTRY FLOW==============
            net = slim.conv2d(inputs, 8, [3, 3], stride=2, padding='same', scope='pool1')
            net = slim.batch_norm(net, scope='pool1_bn1')
            net = tf.nn.relu(net, name='pool1_relu1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='pool2')

            # =========== STAGE ==============
            for i in range(len(layers)):
                net = make_layers(net, layers[i], channels[i], stride=2, scope='stage'+str(i+1))

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['pool3'] = end_points[scope + '/stage1']
            end_points['pool4'] = end_points[scope + '/stage2']
            end_points['pool5'] = net

        return net, end_points


def xception39(inputs, is_training=True, reuse=None, scope='Xception39'):
    layers = [4, 8, 4]
    channels = [16, 32, 64]
    return xception(inputs, layers, channels, is_training=is_training, reuse=reuse, scope=scope)


def xception_arg_scope(weight_decay=0.00001,
                       batch_norm_decay=0.9,
                       batch_norm_epsilon=1e-5):
    '''
    The arg scope for xception model. The weight decay is 1e-5 as seen in the paper.
    INPUTS:
    - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
    - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
    - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.
    OUTPUTS:
    - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
    '''
    # Set weight_decay for weights in conv2d and separable_conv2d layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        activation_fn=None):
        # Set parameters for batch_norm. Note: Do not set activation function as it's preset to None already.
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            return scope