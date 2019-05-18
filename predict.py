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
import cv2
import numpy as np
import time


colors = np.array([[64,128,64],
[192,0,128],
[0,128, 192],
[0, 128, 64],
[128, 0, 0],
[64, 0, 128],
[64, 0, 192],
[192, 128, 64],
[192, 192, 128],
[64, 64, 128],
[128, 0, 192],
[192, 0, 64],
[128, 128, 64],
[192, 0, 192],
[128, 64, 64],
[64, 192, 128],
[64, 64, 0],
[128, 64, 128],
[128, 128, 192],
[0, 0, 192],
[192, 128, 128],
[128, 128, 128],
[64, 128,192],
[0, 0, 64],
[0, 64, 64],
[192, 64, 128],
[128, 128, 0],
[192, 128, 192],
[64, 0, 64],
[192, 192, 0],
[0, 0, 0],
[64, 192, 0]], dtype=np.float32)


if __name__ == '__main__':
    model_config = configuration.MODEL_CONFIG
    train_config = configuration.TRAIN_CONFIG
    infer_size = (736, 960)

    g = tf.Graph()
    with g.as_default():
        # Build the test model
        model = BiseNet(model_config, None, 32, 'inference')
        model.build()
        response = model.response

        saver = tf.train.Saver()
        # Dynamically allocate GPU memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)

        sess = tf.Session(config=sess_config)
        model_path = tf.train.latest_checkpoint(train_config['train_dir'])

        # global_variables_init_op = tf.global_variables_initializer()
        # local_variables_init_op = tf.local_variables_initializer()

        # sess.run(local_variables_init_op)
        saver.restore(sess, model_path)

        # Input prepare
        img = cv2.imread('./example/0001TP_007170.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (infer_size[1], infer_size[0]))
        img = img[np.newaxis, :]

        elapse = []
        for i in range(50):
            start = time.time()
            _ = sess.run(response, feed_dict={model.images_feed: img})
            duration = time.time() - start
            print('time: {:.4f}, about {:.6f} fps'.format(duration, 1 / duration))
            elapse.append(duration)
        print('Average time: {:.4f}, about {:.6f} fps'.format(np.mean(elapse), 1 / np.mean(elapse)))

        predict = tf.reshape(tf.matmul(tf.reshape(tf.one_hot(tf.argmax(response, -1), 32), [-1, 32]), colors),
                             [infer_size[0], infer_size[1], 3])
        predict = sess.run(predict, feed_dict={model.images_feed: img})
        predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Image")
        cv2.imshow('Image', predict)
        cv2.waitKey(0)
        cv2.imwrite('./example/1.png', predict)
