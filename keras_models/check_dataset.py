#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.

v2 - added saver.
"""
	
# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
import numpy as np
import math
import sys, os

sys.path.append('.')
sys.path.append('..')
from tfrecords_converter import TfrecordsDataset

batch_size = 128  # 256
image_shape = (128, 128)
image_channels = 3
dataset = TfrecordsDataset("../dataset/presence_train-bboxes128x128.tfrecords", 
	"../dataset/presence_test-bboxes128x128.tfrecords", 
	image_shape, image_channels, batch_size)
train_dataset = dataset.train_set.batch(batch_size)
valid_dataset = dataset.test_set.batch(batch_size)

graph = tf.Graph()  # сreate a new graph

with graph.as_default():
	
	iterator_train = train_dataset.make_one_shot_iterator()
	next_element_train = iterator_train.get_next()
	iterator_valid = valid_dataset.make_one_shot_iterator()
	next_element_valid = iterator_valid.get_next()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_count = 0
		while True:				
			try:
				features, labels = sess.run(next_element_train)
				train_count += 1
				if train_count % 20 == 0:
					print('train_count', train_count)
					print(labels)
			except tf.errors.OutOfRangeError:
				print("End of training dataset. Count={} batches".format(train_count))
				break	

		valid_count = 0
		while True:				
			try:
				features, labels = sess.run(next_element_valid)
				valid_count += 1	
				if valid_count % 20 == 0:
					print('valid_count', valid_count)								
			except tf.errors.OutOfRangeError:
				print("End of validation dataset. Count={} batches".format(valid_count))
				break	

		print('\nTrain={}, valid={}'.format(train_count, valid_count))