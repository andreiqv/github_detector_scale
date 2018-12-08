import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet	

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2  #224x224.


def model_ResNet50(inputs):

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', 
		input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(1, activation=None)(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model

def model_InceptionV3(inputs):

	base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', 
		input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_InceptionV3')	
	return model
	
def model_MobileNetV2(inputs):
	""" 
	num_layers: 157
	"""

	base_model = MobileNetV2(weights=None, alpha=1.0, depth_multiplier=0.35,
		include_top=False, pooling='avg', input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_MobileNetV2')
	return model	

# -----------------

import tensorflow as tf
from tensorflow import keras

layers = keras.layers

def model1(inputs):
	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		activation='elu',
		use_bias=True)(inputs)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=64,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=128,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(
		filters=64,
		kernel_size=(1, 1),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation=None
	)(x)
	x = layers.Conv2D(
		filters=128,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=256,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(
		filters=5,
		kernel_size=(8, 8),
		strides=(1, 1),
		padding='VALID',
		use_bias=True,
		activation='sigmoid'
	)(x)

	x = layers.Reshape((5,))(x)
	model = keras.Model(inputs, x, name='glp_model1')

	return model


def model2(inputs):
	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		activation='elu',
		use_bias=True)(inputs)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=64,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(
		filters=128,
		kernel_size=(1, 1),
		strides=(1, 1),
		padding='SAME',
		use_bias=True,
		activation=None
	)(x)
	x = layers.Conv2D(
		filters=256,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='elu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=2
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(
		filters=5,
		kernel_size=(7, 7),
		strides=(1, 1),
		padding='VALID',
		use_bias=True,
		activation='sigmoid'
	)(x)

	x = layers.Reshape((5,))(x)
	model = keras.Model(inputs, x, name='glp_model1')

	return model


def darknet_block(filters, kernel, stride, padding, x):
	x = layers.Conv2D(
		filters=filters,
		kernel_size=kernel,
		strides=stride,
		padding=padding,
		activation='tanh',   # 0.7757 -> tanh 0.8070
		use_bias=False)(x)
	#x = layers.LeakyReLU(0.1)(x)
	x = layers.BatchNormalization()(x)
	return x


def model3(inputs):
	x = darknet_block(32, 3, 1, 'SAME', inputs)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(64, 3, 1, 'SAME',  x)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(128, 3, 1, 'SAME', x)
	x = darknet_block(64, 1, 1, 'SAME',  x)
	x = darknet_block(128, 3, 1, 'SAME', x)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(256, 3, 1, 'SAME', x)
	x = darknet_block(64, 3, 2, 'VALID', x)
	x = darknet_block(256, 3, 1, 'SAME', x) # +

	x = layers.Conv2D(
		filters=5,
		kernel_size=(7, 7),
		strides=(1, 1),
		padding='VALID',
		use_bias=False,
		activation='sigmoid'
	)(x)

	x = layers.Reshape((5,))(x)
	model = keras.Model(inputs, x, name='glp_model3')

	return model


def model4(inputs):  # added
	x = darknet_block(32, 3, 1, 'SAME', inputs)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(64, 3, 1, 'SAME',  x)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(128, 3, 1, 'SAME', x)
	x = darknet_block(64, 1, 1, 'SAME',  x)
	x = darknet_block(128, 3, 1, 'SAME', x)
	x = darknet_block(32, 3, 2, 'VALID', x)
	x = darknet_block(256, 3, 1, 'SAME', x)
	x = darknet_block(64, 3, 2, 'VALID', x)
	
	"""
	x = layers.Conv2D(
		filters=5,
		kernel_size=(7, 7),
		strides=(1, 1),
		padding='VALID',
		use_bias=False,
		activation='sigmoid'
	)(x)
	"""

	#x = layers.Reshape((5,))(x)

	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='elu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)

	model = keras.Model(inputs, x, name='glp_model3')

	return model

#------

def model_first(inputs):
	x = layers.Conv2D(
		filters=8,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='VALID',
		activation='relu',
		use_bias=True)(inputs)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=16,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='relu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=16,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='relu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='relu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='VALID',
		use_bias=True,
		activation='relu'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='elu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)

	model = keras.Model(inputs, x, name='first_glp_model')

	return model


def model_first2(inputs):
	x = layers.Conv2D(
		filters=8,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='VALID',
		activation='tanh',
		use_bias=True)(inputs)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=16,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='tanh'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=16,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='tanh'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(2, 2),
		padding='VALID',
		use_bias=True,
		activation='tanh'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)

	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		strides=(1, 1),
		padding='VALID',
		use_bias=True,
		activation='tanh'
	)(x)
	x = layers.MaxPool2D(
		pool_size=2,
		strides=1
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='sigmoid')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1, activation='sigmoid')(x)
	#x = layers.Dense(5, activation=None)(x)

	model = keras.Model(inputs, x, name='model_first2')

	return model

