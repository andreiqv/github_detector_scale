import tensorflow as tf
slim = tf.contrib.slim
from settings import IMAGE_SIZE

def cnn(inputs, num_classes=1000, is_training=True):

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0]*IMAGE_SIZE[1]*3])  # 128 x 128 x 3
	net = slim.conv2d(images, 8, [5,5], scope='conv1')   
	net = slim.max_pool2d(net, [2,2], scope='pool1')  # 64 x 64  x 8
	net = slim.conv2d(net, 16, [5,5], scope='conv2')
	net = slim.max_pool2d(net, [2,2], scope='pool2')  # 32 x 32 x 16
	net = slim.conv2d(net, 16, [5,5], scope='conv3')
	net = slim.max_pool2d(net, [2,2], scope='pool3')  # 16 x 16 x 16
	net = slim.conv2d(net, 16, [5,5], scope='conv3')
	net = slim.max_pool2d(net, [2,2], scope='pool3')  # 8 x 8 x 16
	net = slim.conv2d(net, 32, [5,5], scope='conv3')
	net = slim.max_pool2d(net, [2,2], scope='pool3')  # 4 x 4 x 32
	net = slim.flatten(net, scope='flatten3')
	#net = slim.fully_connected(net, 500, scope='fc4')	
	logits = slim.fully_connected(x, num_classes, scope='fc/fc_1')
	end_points = ['none']
	return logits, end_points

