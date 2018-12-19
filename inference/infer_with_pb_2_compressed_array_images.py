#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Using TF for inference, and TensorRT for compress a graph.

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import timer

from time import sleep
import io
from picamera import PiCamera
camera = PiCamera()
stream = io.BytesIO()

#import tensorflow.contrib.tensorrt as trt


use_hub_model = False

if True:
	FROZEN_FPATH = '/home/pi/work/pb/model_first_3-60-1.000-1.000[0.803].pb'
	#FROZEN_FPATH = '/home/pi/work/pb/model_resnet50-97-0.996-0.996[0.833].pb'
	ENGINE_FPATH = 'saved_model_full_2.plan'
	INPUT_SIZE = [3, 128, 128]
	INPUT_NODE = 'input_1'
	OUTPUT_NODE = 'dense_1/Sigmoid'
	#OUTPUT_NODE = 'dense/Sigmoid'
	input_output_placeholders = [INPUT_NODE + ':0', OUTPUT_NODE + ':0']



def get_image_as_array(image_file):

	# Read the image & get statstics
	image = Image.open(image_file)
	#img.show()
	#shape = [299, 299]
	shape = tuple(INPUT_SIZE[1:])
	#image = tf.image.resize_images(img, shape, method=tf.image.ResizeMethod.BICUBIC)
	image = image.resize(shape, Image.ANTIALIAS)
	image_arr = np.array(image, dtype=np.float32) / 255.0

	return image_arr


def get_labels(labels_file):	

	with open(labels_file) as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
		print(labels)
	#sys.exit(0)
	return labels


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
		max_workspace_size_bytes=2<<20,
		precision_mode=precision_mode)

	return trt_graph


def inference_with_graph(graph_def, image):
	""" Predict for single images
	"""

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Import a graph_def into the current default Graph
			print("import graph")	
			input_, predictions =  tf.import_graph_def(graph_def, name='', 
				return_elements=input_output_placeholders)
			
			camera.start_preview()

			timer.timer('predictions.eval')			

			time_res = []
			for i in range(10):

				camera.capture(stream, format='jpeg')
				stream.seek(0)
				image = Image.open(stream)
				shape = tuple(INPUT_SIZE[1:])
				image = image.resize(shape, Image.ANTIALIAS)
				image_arr = np.array(image, dtype=np.float32) / 255.0				

				pred_val = predictions.eval(feed_dict={input_: [image_arr]})
				print(pred_val)
				timer.timer()
				#time_res.append(0)
				#print('index={0}, label={1}'.format(index, label))

			camera.stop_preview()	

			#print('mean time = {0}'.format(np.mean(time_res)))

			#return index




def inference_images_with_graph(graph_def, filenames):
	""" Process list of files of images.
	"""

	images = [get_image_as_array(filename) for filename in filenames]

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Import a graph_def into the current default Graph
			print("import graph")	
			input_, predictions =  tf.import_graph_def(graph_def, name='', 
				return_elements=input_output_placeholders)
			
			camera.start_preview()

			for i in range(len(filenames)):
				filename = filenames[i]
				image = images[i]

				p_val = predictions.eval(feed_dict={input_: [image]})
				index = np.argmax(p_val)
				label = labels[index]

				print('{0}: prediction={1}'.format(filename, label))

			camera.stop_preview()	

def createParser ():
	"""ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default=None, type=str,\
		help='input')
	parser.add_argument('-dir', '--dir', default="../images", type=str,\
		help='input')	
	parser.add_argument('-pb', '--pb', default="saved_model.pb", type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])		
	pb_file = arguments.pb

	if arguments.input is not None:
		filenames = [arguments.input]
		#image = get_image_as_array(filename)
		#images = [(image]

	else:
		filenames = []
		src_dir = arguments.dir
		listdir = os.listdir(src_dir)
		for f in listdir:
			filenames.append(src_dir + '/' + f)

	assert type(filenames) is list and filenames != []


	#labels = get_labels('labels.txt')
	pb_file = FROZEN_FPATH
	graph_def = get_frozen_graph(pb_file)

	#modes = ['FP32', 'FP16', 0]
	#precision_mode = modes[2]

	#pb_file_name = 'saved_model.pb' # output_graph.pb

	# no compress
	image_file = '/home/pi/work/images/img_1_0_2018-08-04-09-37-300672_5.jpg'
	image = get_image_as_array(image_file)
	inference_with_graph(graph_def, image)
	#inference_images_with_graph(graph_def, filenames)		


	"""
	for mode in modes*2:
		print('\nMODE: {0}'.format(mode))
		graph_def = compress_graph_with_trt(graph_def, mode)
		inference_with_graph(graph_def, images, labels)
	"""		

"""
0.0701 sec. (total time 1.72) - model_first_3-60-1.000-1.000[0.803].pb

0.7628 sec. -- model_resnet50-97-0.996-0.996[0.833].pb

"""