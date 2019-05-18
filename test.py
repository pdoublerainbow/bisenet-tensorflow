#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#


"""predict the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.bisenet import BiseNet
import configuration
import logging

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    model_config = configuration.MODEL_CONFIG
    train_config = configuration.TRAIN_CONFIG

    g = tf.Graph()
    with g.as_default():
        # Build the test model
        model = BiseNet(model_config, train_config, 32, 'test')
        model.build()

        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(train_config['test_data_config']['test_dir'], g)
        summary_op = tf.summary.merge_all()
        # Dynamically allocate GPU memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)

        sess = tf.Session(config=sess_config)
        model_path = tf.train.latest_checkpoint(train_config['train_dir'])

        config = train_config['test_data_config']
        total_steps = config['num_examples_per_epoch']//config['batch_size']
        logging.info('Train for {} steps'.format(total_steps))

        # global_variables_init_op = tf.global_variables_initializer()
        local_variables_init_op = tf.local_variables_initializer()

        sess.run(local_variables_init_op)
        saver.restore(sess, model_path)

        for step in range(total_steps):
            predict_loss, loss = sess.run([model.loss, model.total_loss])
            format_str = 'step %d, total loss = %.2f, predict loss = %.2f'
            logging.info(format_str % (step, loss, predict_loss))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
