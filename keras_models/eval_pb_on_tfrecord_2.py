#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Скрипт для валидации модели в виде pb-файла.

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
#import timer

sys.path.append('.')
sys.path.append('..')
from keras_models.aux import miou, bboxes_loss, accuracy
from tfrecords_converter import TfrecordsDataset
#import tensorflow.contrib.tensorrt as trt


FROZEN_FPATH = './output/model_resnet50-97-0.996-0.996[0.833].pb'
#ENGINE_FPATH = 'saved_model_full_2.plan'
INPUT_SIZE = [3, 299, 299]
INPUT_NODE = 'input_1'
OUTPUT_NODE = 'dense/Sigmoid'	
input_output_placeholders = ['input_1:0', 'dense/Sigmoid:0']

BATCH_SIZE = 1  # 256


def get_frozen_graph(pb_file):

	# We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
	with gfile.FastGFile(pb_file,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		#sess.graph.as_default() #new line	
	return graph_def


def compress_graph_with_trt(graph_def, precision_mode):

	output_node = input_output_placeholders[1]

	if precision_mode==0: 
		return graph_def

	trt_graph = trt.create_inference_graph(
		graph_def,
		[output_node],
		max_batch_size=1,
		max_workspace_size_bytes = 2<<20,
		precision_mode=precision_mode)

	return trt_graph


def evaluate_pb_model(graph_def, dataset):
	""" 
	"""
	train_steps_per_epoch = 10 #1157
	valid_steps_per_epoch = 10  #77
	train_dataset = dataset.train_set.repeat()
	valid_dataset = dataset.test_set.batch(BATCH_SIZE).repeat()

	with tf.Graph().as_default() as graph:

		iterator_train = train_dataset.make_one_shot_iterator()
		next_element_train = iterator_train.get_next()
		iterator_valid = valid_dataset.make_one_shot_iterator()
		next_element_valid = iterator_valid.get_next()		

		with tf.Session() as sess:

			# Import a graph_def into the current default Graph
			print("import graph")	
			input_, logits_ =  tf.import_graph_def(graph_def, name='', 
				return_elements=input_output_placeholders)
			#labels = tf.Variable()			
			labels_ = tf.placeholder(tf.float32, [None, 5], name='labels')
			miou_ = miou(labels_, logits_)

			for i in range(train_steps_per_epoch):
				features, labels = sess.run(next_element_train)

				predict_values = logits_.eval(feed_dict={input_: [features]})
				#index = np.argmax(p_val)
				#label = labels[index]
				print('labels:')
				print(labels)
				print('predictions:')
				print(predict_values)
				#miou_value = miou(labels, predict_values)
				miou_value = miou_.eval(feed_dict={input_: [features], labels_:[labels]})
				#print('miou:', miou_value)
				print()

				#print('{0}: prediction={1}'.format(filename, label))



def createParser ():
	"""ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	#parser.add_argument('-i', '--input', default=None, type=str,\
	#	help='input')
	#parser.add_argument('-dir', '--dir', default="../images", type=str,\
	#	help='input')	
	parser.add_argument('-pb', '--pb', default=None, type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])
	
	if arguments.pb:
		pb_file = arguments.pb
	else:
		pb_file = FROZEN_FPATH


	image_shape = (128, 128)
	image_channels = 3
	dataset = TfrecordsDataset("../dataset/train-bboxes128x128.tfrecords", 
		"../dataset/test-bboxes128x128.tfrecords", 
		image_shape, image_channels, BATCH_SIZE)


	graph_def = get_frozen_graph(pb_file)
	evaluate_pb_model(graph_def, dataset)		


	"""
	for mode in modes*2:
		print('\nMODE: {0}'.format(mode))
		graph_def = compress_graph_with_trt(graph_def, mode)
		inference_with_graph(graph_def, images, labels)
	"""		