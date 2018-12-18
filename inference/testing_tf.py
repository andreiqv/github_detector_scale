#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TF work testing """

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
# Import the training data (MNIST)
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))