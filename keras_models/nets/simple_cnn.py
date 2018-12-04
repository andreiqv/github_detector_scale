import tensorflow as tf
slim = tf.contrib.slim
from settings import IMAGE_SIZE

def cnn_3(inputs, num_classes=1000, is_training=True):

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])  # 128 x 128 x 3
	x = slim.conv2d(x, 16, [5,5], scope='conv1')   
	x = slim.flatten(x, scope='flatten3')
	x = slim.fully_connected(x, 1000, activation_fn=tf.nn.sigmoid, scope='fc_hid')	
	logits = slim.fully_connected(x, num_classes, activation_fn=None, scope='fc_last')
	end_points = ['none']
	return logits, end_points


def cnn_2(inputs, num_classes=1000, is_training=True):

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])  # 128 x 128 x 3
	x = slim.conv2d(x, 16, [5,5], scope='conv1')   
	x = slim.max_pool2d(x, [2,2], scope='pool1')  # 64 x 64  x 
	x = slim.conv2d(x, 32, [5,5], scope='conv2')
	x = slim.max_pool2d(x, [2,2], scope='pool2')  # 32 x 32 x 
	x = slim.conv2d(x, 32, [5,5], scope='conv3')
	x = slim.max_pool2d(x, [2,2], scope='pool3')  # 16 x 16 x 
	x = slim.conv2d(x, 32, [5,5], scope='conv4')
	x = slim.max_pool2d(x, [2,2], scope='pool4')  # 8 x 8 x 32
	x = slim.flatten(x, scope='flatten3')
	x = slim.fully_connected(x, 200, activation_fn=tf.nn.sigmoid, scope='fc_hid')	
	logits = slim.fully_connected(x, num_classes, activation_fn=None, scope='fc_last')
	end_points = ['none']
	return logits, end_points


def cnn_1(inputs, num_classes=1000, is_training=True):

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])  # 128 x 128 x 3
	x = slim.conv2d(x, 8, [5,5], scope='conv1')   
	x = slim.max_pool2d(x, [2,2], scope='pool1')  # 64 x 64  x 8
	x = slim.conv2d(x, 16, [5,5], scope='conv2')
	x = slim.max_pool2d(x, [2,2], scope='pool2')  # 32 x 32 x 16
	x = slim.conv2d(x, 16, [5,5], scope='conv3')
	x = slim.max_pool2d(x, [2,2], scope='pool3')  # 16 x 16 x 16
	x = slim.conv2d(x, 16, [5,5], scope='conv4')
	x = slim.max_pool2d(x, [2,2], scope='pool4')  # 8 x 8 x 16
	x = slim.conv2d(x, 32, [5,5], scope='conv5')
	x = slim.max_pool2d(x, [2,2], scope='pool5')  # 4 x 4 x 32
	x = slim.flatten(x, scope='flatten3')
	#x = slim.fully_connected(x, 500, scope='fc4')	
	logits = slim.fully_connected(x, num_classes, scope='fc1')
	end_points = ['none']
	return logits, end_points

