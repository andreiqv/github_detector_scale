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
import argparse

sys.path.append('.')
sys.path.append('..')

# tf.enable_eager_execution()
#import settings
from settings import IMAGE_SIZE
from keras_models.utils.timer import timer
#from augment import images_augment

#--------------------

slim = tf.contrib.slim

#import models.inception_v3 as inception
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet
from keras_models.nets import mobilenet_v1
from keras_models.nets import inception_v4
from keras_models.nets.mobilenet import mobilenet_v2
from keras_models.nets.nasnet import nasnet

#-----------------
# Select network

#from nets import simple_fc
#net, net_model_name = simple_fc.fc, 'simple_fc'
from nets import simple_cnn
#net, net_model_name = simple_cnn.cnn_3, 'simple_cnn_3'
#net, net_model_name = simple_cnn.fc2, 'simple_fc2'
#net, net_model_name = alexnet.alexnet_v2, 'alexnet_v2'
#net, net_model_name = inception_v4.inception_v4, 'inception_v4'
net, net_model_name = resnet_v2.resnet_v2_50, 'resnet_v2_50'
#net, net_model_name = resnet_v2.resnet_v2_152, 'resnet_v2_152'
#net, net_model_name = mobilenet_v2.mobilenet_v2_050, 'mobilenet_v2_050'
#net, net_model_name = mobilenet_v2.mobilenet_v2_035, 'mobilenet_v2_035'


#net = inception.inception_v3
#net = inception.inception_v4
#net = vgg.vgg_19
#net = mobilenet_v1.mobilenet_v1
#net = mobilenet_v2.mobilenet_v2_035
#net = nasnet.build_nasnet_mobile

#net_model_name = 'resnet_v2_152'
#net = resnet_v2.resnet_v2_152

#--------------
DEBUG = False

OUTPUT_NODE = 'softmax'
num_classes = 1
print('num_classes:', num_classes)
#print('IMAGE_SIZE:', IMAGE_SIZE) #IMAGE_SIZE = (299, 299) 
print('Network name:', net_model_name)

#--
# for saving results
results = {'epoch':[], 'train_loss':[], 'valid_loss':[], 'train_acc':[],\
	'valid_acc':[], 'train_top6':[], 'valid_top6':[]}
results_filename = '_results_{}.txt'.format(net_model_name)
f_res = open(results_filename, 'wt')
dir_for_pb = 'pb'
dir_for_checkpoints = 'checkpoints'
checkpoint_name = net_model_name
os.system('mkdir -p {}'.format(dir_for_pb))
os.system('mkdir -p {}'.format(dir_for_checkpoints))


#--
# plotting
SHOW_PLOT = True
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.suptitle(net_model_name, fontsize=16)

def plot_figure(results, ax1, ax2):
	ax1.cla()
	ax1.plot(results['epoch'], results['train_loss'])
	ax1.plot(results['epoch'], results['valid_loss'])
	ax1.legend(['train_loss', 'valid_loss'], loc='upper right')
	ax1.grid(color='g', linestyle='-', linewidth=0.2)
	ax1.set_ylim(0, 0.5)
	ax2.cla()
	ax2.plot(results['epoch'], results['train_acc'])
	ax2.plot(results['epoch'], results['valid_acc'])
	ax2.legend(['train_acc', 'valid_acc'], loc='upper left')
	ax2.grid(color='g', linestyle='-', linewidth=0.2)
	ymaxval = max(results['valid_acc'])
	ymin = 0.95 if ymaxval > 0.98 else (0.9 if ymaxval > 0.95 else 0.8)
	ax2.set_ylim(ymin, 1.0)
	#plt.show()
	outfile = '_plot_[{}].png'.format(net_model_name)
	plt.savefig(outfile)


#------------
# dataset
from tfrecords_converter_regression import TfrecordsDataset
batch_size = 128  # 256
image_shape = (128, 128)
image_channels = 3
dataset = TfrecordsDataset("../dataset/regression_train-bboxes128x128.tfrecords", 
	"../dataset/regression_test-bboxes128x128.tfrecords", 
	image_shape, image_channels, batch_size)

AUGMENT = True
if AUGMENT:
	dataset.augment_train_dataset()
	train_dataset = dataset.train_set #.batch(batch_size)
else:
	train_dataset = dataset.train_set.batch(batch_size)
valid_dataset = dataset.test_set.batch(batch_size)

num_epochs = 500		
epochs_checkpoint = 100 # interval for saving checkpoints and pb-file 
train_steps_per_epoch = 469 #1157
valid_steps_per_epoch = 24  #77
train_dataset = train_dataset.repeat()
valid_dataset = valid_dataset.repeat()

"""
def model_function(next_element):
	x, y = next_element
	logits, end_points = inception.inception_v3(
		x, num_classes=settings.num_classes, is_training=True)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	return logits, loss
"""

def calc_mean_acc(labels, outputs):

	th = 0.5
	#results = np.map(lambda x: 1 if x[0][0] > th else 0, train_outputs)
	#np.mean(train_loss)
	vf = np.vectorize(lambda x: 1.0 if x > th else 0.0)
	results = list(map(lambda x: x[0], outputs))
	results = vf(results)
	labels = list(map(lambda x: x[0], labels))
	coincidence = 1 - np.abs(np.array(results) - np.array(labels))
	#print(labels)
	#print(results)
	#print(coincidence)
	#sys.exit()
	return np.mean(coincidence)


def createParser ():
	"""	ArgumentParser """
	parser = argparse.ArgumentParser()
	#parser.add_argument('-r', '--restore', dest='restore', action='store_true')
	parser.add_argument('-rc', '--restore_checkpoint', default=None, type=str, help='Restore from checkpoints')
	return parser

if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])	

	graph = tf.Graph()  # сreate a new graph

	with graph.as_default():
		
		iterator_train = train_dataset.make_one_shot_iterator()
		next_element_train = iterator_train.get_next()
		iterator_valid = valid_dataset.make_one_shot_iterator()
		next_element_valid = iterator_valid.get_next()

		#iterator_train = train_dataset.make_initializable_iterator()
		#x, y = next_element_train

		x = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input')
		y = tf.placeholder(tf.float32, [None, 1], name='y')

		#x = images_augment(x)

		#with tf.device("/device:GPU:1"):
		logits, end_points = net(x, num_classes=num_classes, is_training=True)
		variables_to_restore = slim.get_variables_to_restore()
		output = tf.reshape(logits, [-1, 1])
		#output = tf.nn.softmax(logits, name=OUTPUT_NODE)

		loss = tf.reduce_mean(tf.squared_difference(y, output))
		#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		train_op = tf.train.AdagradOptimizer(0.005).minimize(loss)
		#train_op = tf.train.AdamOptimizer(0.02).minimize(loss)		
		#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
		
		acc = loss
		acc_top6 = loss
		#acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1 - mean value	
		#acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)  # list values for batch.
				

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			if arguments.restore_checkpoint is not None:			
				
				tf.train.Saver(variables_to_restore).restore(sess, './{}/{}'.\
					format(dir_for_checkpoints, arguments.restore_checkpoint))			

			for epoch in range(num_epochs):
				print('\nEPOCH {}/{} ({})'.format(epoch, num_epochs, net_model_name))

				timer('train, epoch {0}'.format(epoch))
				train_loss_list, train_acc_list, train_top6_list = [], [], []

				for i in range(train_steps_per_epoch):
					
					try:
						features, labels = sess.run(next_element_train)
						
						sess.run(train_op, feed_dict={x: features, y: labels})
						
						#train_acc, train_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})
						train_outputs, train_loss, train_acc, train_top6 = sess.run([output, loss, acc, acc_top6], feed_dict={x: features, y: labels})

						#print('train:', i, labels[0], train_logits[0])

						"""
						th = 0.5
						#results = np.map(lambda x: 1 if x[0][0] > th else 0, train_outputs)
						#np.mean(train_loss)
						vf = np.vectorize(lambda x: 1.0 if x > th else 0.0)
						results = list(map(lambda x: x[0], train_outputs))
						results = vf(results)
						labels = list(map(lambda x: x[0], labels))
						coincidence = np.abs(1 - np.array(results) + np.array(labels))
						#print(labels)
						#print(results)
						#print(coincidence)
						#sys.exit()
						train_acc = np.mean(coincidence)
						"""

						train_acc = calc_mean_acc(labels, train_outputs)

						train_loss_list.append(np.mean(train_loss))
						train_acc_list.append(train_acc)
						train_top6_list.append(np.mean(train_top6))

						if i % 30 == 0:

							if False:
								for j in range(len(labels)):
									if  labels[j][0] !=  train_outputs[j][0]:
										print('train:', i, j, labels[j], train_outputs[j])

							timer('epoch={} i={}: train loss={:.4f}, acc={:.4f}'.\
								format(epoch, i, np.mean(train_loss_list), 
								np.mean(train_acc_list))) # np.mean(train_top6_list)

						
					except tf.errors.OutOfRangeError:
						print("End of training dataset.")
						break	


				# valid
				timer('valid, epoch {0}'.format(epoch))
				valid_loss_list = []
				valid_acc_list = []
				valid_top6_list = []			

				for i in range(valid_steps_per_epoch):
					
					try:
						features, labels = sess.run(next_element_valid)
						valid_outputs, valid_loss, valid_acc, valid_top6 = sess.run([output, loss, acc, acc_top6], feed_dict={x: features, y: labels})
						
						valid_acc = calc_mean_acc(labels, valid_outputs)

						valid_loss_list.append(np.mean(valid_loss))
						valid_acc_list.append(valid_acc)
						valid_top6_list.append(np.mean(valid_top6))

							#print('valid:', i, labels[0], valid_logits[0])

						if i % 20 == 0:

							if True:
								for j in range(len(labels)):
									if labels[j][0] != valid_outputs[j][0]:
										print('valid:', i, j, labels[j], valid_outputs[j])

							print('epoch={} i={}: valid acc={:.4f}'.\
								format(epoch, i, np.mean(valid_acc_list)))

					except tf.errors.OutOfRangeError:
						print("End of valid dataset.")
						break			
				timer()

				# result for each epoch
				mean_train_loss = np.mean(train_loss_list)
				mean_valid_loss = np.mean(valid_loss_list)
				mean_train_acc = np.mean(train_acc_list)
				mean_valid_acc = np.mean(valid_acc_list)
				mean_train_top6 = np.mean(train_top6_list)
				mean_valid_top6 = np.mean(valid_top6_list)
				res = '[{:02}]: TRAIN loss={:.4f} acc={:.4f}, VALID loss={:.4f} acc={:.4f}\n'.\
					format(epoch, mean_train_loss, mean_train_acc,
						mean_valid_loss, mean_valid_acc)
				#res = '[{:02}]: TRAIN loss={:.4f} acc={:.4f} top6={:.4f}; VALID loss={:.4f} acc={:.4f} top6={:.4f}\n'.\
				#	format(epoch, mean_train_loss, mean_train_acc, mean_train_top6,
				#		mean_valid_loss, mean_valid_acc, mean_valid_top6)
				print(res)
				f_res.write(res)
				f_res.flush()

				results['epoch'].append(epoch)
				results['train_loss'].append(mean_train_loss)
				results['valid_loss'].append(mean_valid_loss)
				results['train_acc'].append(mean_train_acc)
				results['valid_acc'].append(mean_valid_acc)
				results['train_top6'].append(mean_train_top6)
				results['valid_top6'].append(mean_valid_top6)			
				if SHOW_PLOT:
					plot_figure(results, ax1, ax2)
					#_thread.start_new_thread(plot_figure, ())

				if epoch % epochs_checkpoint == 0 and epoch > 1:
					# save_checkpoints	
					saver = tf.train.Saver(variables_to_restore)		
					saver.save(sess, './{}/{}'.\
						format(dir_for_checkpoints, checkpoint_name))  

					# SAVE GRAPH TO PB
					graph = sess.graph			
					tf.graph_util.remove_training_nodes(graph.as_graph_def())
					# tf.contrib.quantize.create_eval_graph(graph)
					# tf.contrib.quantize.create_training_graph()

					output_node_names = [OUTPUT_NODE]
					output_graph_def = tf.graph_util.convert_variables_to_constants(
						sess, graph.as_graph_def(), output_node_names)
					# save graph:		
					pb_file_name = '{}_(ep={}_acc={:.4f}_top6={:.4f}).pb'.format(net_model_name, epoch, mean_valid_acc, mean_valid_top6)
					tf.train.write_graph(output_graph_def, dir_for_pb, pb_file_name, as_text=False)	
	# end of training
	f_res.close()
