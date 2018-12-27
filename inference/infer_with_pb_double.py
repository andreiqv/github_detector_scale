#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
On raspberry:
- resnet18: 0.3741 sec.
- model_first_3: 0.0664 sec.

Resnet on SERVER: GPU: 0.0227 sec., CPU: 0.0441 sec.
"""

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import timer

from time import sleep
import random
import io

USE_CAMERA = False # capture images from camera
if USE_CAMERA:
	from picamera import PiCamera
	from picamera.array import PiRGBArray
	# initialize the camera and grab a reference to the raw camera capture
	camera = PiCamera()
	camera.resolution = (640, 480)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(640, 480))

#import tensorflow.contrib.tensorrt as trt


use_hub_model = False

if True:
	PB1_PATH = '../pb/model_test_64_v6-389-0.994-0.994[0.718].pb'    # 0.0120 sec.	
	#PB2_PATH = '../pb/presence_model_3_2-95-1.000-1.000[0.818].pb'   # 0.0731 sec.
	#PB2_PATH = '../pb/model_first_3-102-1.000-1.000[0.808].pb'      # 0.0502 sec.
	#PB2_PATH = '../pb/presence_resnet18_2-59-1.000-1.000[0.825].pb' # 0.3639 sec. # 'input_1' and 'dense/Sigmoid'
	#PB2_PATH = '../pb/model_resnet50-97-0.996-0.996[0.833].pb'   # 0.0731 sec.
	#PB2_PATH = '../pb/presence_model_3.pb' 
	PB2_PATH = '../pb/model_3_2-165-1.000-1.000[0.818].pb'

	INPUT_SIZE_1 = [3, 64, 64]
	INPUT_SIZE_2 = [3, 128, 128]
	INPUT_NODE_1 = 'input'
	INPUT_NODE_2 = 'input' # 'input_1'
	OUTPUT_NODE_1 = 'output/Sigmoid'
	OUTPUT_NODE_2 = 'output/Sigmoid' # 'dense/Sigmoid'	 
	input_output_placeholders_1 = [INPUT_NODE_1 + ':0', OUTPUT_NODE_1 + ':0']
	input_output_placeholders_2 = [INPUT_NODE_2 + ':0', OUTPUT_NODE_2 + ':0']



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


def load_image(image_file):
	return Image.open(image_file)

def image_to_array(image):
	return np.array(image, dtype=np.float32) / 255.0


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



def inference_FROM_CAMERA_with_SINGLE_graph(graph_def):
	""" Predict for single images.
	Use single nn model.
	"""

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Import a graph_def into the current default Graph
			print("import graph")	
			#input_, predictions =  tf.import_graph_def(graph_def, name='', 
			#	return_elements=input_output_placeholders)
			
			#camera.start_preview()
			#camera.resolution = (640, 480)
			#camera.framerate = 32

			timer.timer('predictions.eval')	

			time_res = []
			for frame in camera.capture_continuous(\
									rawCapture, format="bgr", use_video_port=True):
				# grab the raw NumPy array representing the image - this array
				# will be 3D, representing the width, height, and # of channels
				image_arr = frame.array
				# clear the stream in preparation for the next frame
				rawCapture.truncate(0)			

				image_cam = Image.fromarray(np.uint8(image_arr))				
				shape = tuple(INPUT_SIZE[1:])
				image = image_cam.resize(shape, Image.ANTIALIAS)
				image_arr = np.array(image, dtype=np.float32) / 255.0				

				input_, predictions =  tf.import_graph_def(graph_def, name='', 
					return_elements=input_output_placeholders)

				pred_values = predictions.eval(feed_dict={input_: [image_arr]})
				pred = pred_values[0]
				print(pred)
				timer.timer()
				#time_res.append(0)
				#print('index={0}, label={1}'.format(index, label))

				sx, sy = image_cam.size
				x = pred[0] * sx
				y = pred[1] * sy
				w = pred[2] * sx
				h = pred[3] * sy
				box = (x, y, w, h)
				crop = image.crop(box)
				crop.save('crop_{:010d}.jpg'.format(random.randint(0,1000000)), 'jpeg')
				#sys.exit()

			#camera.stop_preview()	
			print(camera.resolution)

			#print('mean time = {0}'.format(np.mean(time_res)))

			#return index




def inference_with_two_graphs(graph_def_1, graph_def_2, image):
	""" 
	image : PIL-image
	"""

	image1 = image.resize(tuple(INPUT_SIZE_1[1:]), Image.ANTIALIAS)
	image1_arr = image_to_array(image1)
	image2 = image.resize(tuple(INPUT_SIZE_2[1:]), Image.ANTIALIAS)
	image2_arr = image_to_array(image2)

	graph1 = tf.Graph()
	sess1 = tf.Session(graph=graph1)
	graph2 = tf.Graph()
	sess2 = tf.Session(graph=graph2)

	with graph1.as_default() as graph:
		print("import graph 1")
		#with sess1 as sess:
		inputs1, predictions1 =  tf.import_graph_def(graph_def_1, name='g1', 
			return_elements=input_output_placeholders_1)
			
	with graph2.as_default() as graph:
		print("import graph 2")
		#with sess2 as sess:
		inputs2, predictions2 =  tf.import_graph_def(graph_def_2, name='g2', 
			return_elements=input_output_placeholders_2)

	timer.timer('PB1')
	pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})	
	timer.timer('PB1')
	pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})	
	timer.timer('PB1')
	pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})	
	timer.timer('PB1')
	pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})	

	pred = pred_values1[0]
	print('PB1:', pred)

	THRESHOLD = 0.7
	if pred[4] > THRESHOLD:
		timer.timer('PB2')
		pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image2_arr]})
		timer.timer('PB2')
		pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image2_arr]})
		timer.timer('PB2')
		pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image2_arr]})

		pred = pred_values2[0]
		print('PB2:', pred)
		timer.timer()
		
		return pred[:4]
	else:
		return None
	

	"""
	pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image_arr]})
	pred = pred_values1[0]
	print('PB1:', pred)
	timer.timer()

	pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image_arr]})
	pred = pred_values2[0]
	print('PB2:', pred)
	timer.timer()
	"""

	return pred



def inference_from_camera_with_two_graphs(graph_def_1, graph_def_2):
	""" Capture images from camera.
	"""

	graph1 = tf.Graph()
	sess1 = tf.Session(graph=graph1)
	graph2 = tf.Graph()
	sess2 = tf.Session(graph=graph2)

	with graph1.as_default() as graph:
		print("import graph 1")
		#with sess1 as sess:
		inputs1, predictions1 =  tf.import_graph_def(graph_def_1, name='g1', 
			return_elements=input_output_placeholders_1)
			
	with graph2.as_default() as graph:
		print("import graph 2")
		#with sess2 as sess:
		inputs2, predictions2 =  tf.import_graph_def(graph_def_2, name='g2', 
			return_elements=input_output_placeholders_2)

	timer.timer('predictions.eval')	
	time_res = []
	for i, frame in enumerate(camera.capture_continuous(\
							rawCapture, format="bgr", use_video_port=True)):
		# grab the raw NumPy array representing the image - this array
		# will be 3D, representing the width, height, and # of channels
		image_arr = frame.array

		image_cam = Image.fromarray(np.uint8(image_arr))				
		image1 = image_cam.resize(tuple(INPUT_SIZE_1[1:]), Image.ANTIALIAS)
		image1_arr = np.array(image1, dtype=np.float32) / 255.0
		if tuple(INPUT_SIZE_1[1:]) == tuple(INPUT_SIZE_2[1:]):
			image2_arr = image1_arr
		else:
			image2 = image_cam.resize(tuple(INPUT_SIZE_2[1:]), Image.ANTIALIAS)
			image2_arr = np.array(image2, dtype=np.float32) / 255.0

		pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})
		pred = pred_values1[0]
		print(pred)
		timer.timer()

		pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image2_arr]})
		pred = pred_values2[0]
		print(pred)
		timer.timer()		
		#time_res.append(0)
		#print('index={0}, label={1}'.format(index, label))

		sx, sy = image_cam.size
		x = pred[0] * sx
		y = pred[1] * sy
		w = pred[2] * sx
		h = pred[3] * sy
		box = (x, y, w, h)
		crop = image.crop(box)
		#crop.save('crop.jpg', 'jpeg')
		#sys.exit()

		#camera.stop_preview()	
		#print(camera.resolution)

		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)		
		sys.exit()


"""
def inference_images_with_graph(graph_def, filenames):
	# Process list of files of images.
	
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
"""

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
	graph_def_1 = get_frozen_graph(PB1_PATH)
	graph_def_2 = get_frozen_graph(PB2_PATH)

	#modes = ['FP32', 'FP16', 0]
	#precision_mode = modes[2]

	if USE_CAMERA:
		inference_from_camera_with_two_graphs(graph_def_1, graph_def_2)

	else:
		files = ['../images/01.jpg', '../images/02.jpg']
		for image_file in files:		
			print(image_file)			
			image = load_image(image_file)
			pred = inference_with_two_graphs(graph_def_1, graph_def_2, image)
			if pred is not None:
				image_for_classificator = image.resize((299, 299), Image.ANTIALIAS)
				sx, sy = image_for_classificator.size
				x = int(pred[0] * sx)
				y = int(pred[1] * sy)
				w = int(pred[2] * sx)
				h = int(pred[3] * sy)
				#w = min(w, 2*(sx-x), 2*x)
				#h = min(h, 2*(sy-y), 2*y)
				print('sx, sy = ',(sx,sy))
				print('x, y = ',(x,y))
				print('w, h = ',(w,h))
				x0 = max(0,  x - w//2)
				x1 = min(sx, x + w//2)
				y0 = max(0,  y - h//2)
				y1 = min(sy, y + h//2)				
				box = (x0, y0, x1, y1)
				crop = image_for_classificator.crop(box)
				crop.save('crop_{:010d}.jpg'.format(random.randint(0,1000000)), 'jpeg')	

