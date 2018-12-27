import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.keras import regularizers

slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet	
from nets import mobilenet_v1
from nets import inception_v4
from nets.mobilenet import mobilenet_v2
from nets.nasnet import nasnet


OUTPUT_NAME = 'output'

def conv(x, f, k, s=1, p='SAME', a='tanh'):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding=p,
		activation=a, # relu, selu
		#kernel_regularizer=regularizers.l2(0.01),
		use_bias=True)(x)
	return x
maxpool = lambda x, p=2, s=1: layers.MaxPool2D(pool_size=p, strides=s)(x)
maxpool2 = lambda x, p=2: layers.MaxPool2D(pool_size=p)(x)	
bn = lambda x: layers.BatchNormalization()(x)


def model_cnn_128(inputs):
	x = inputs 
	x = maxpool(x)  # 64
	x = maxpool(x)  # 32
	x = maxpool(x)  # 16
	x = maxpool(x)  # 8
	x = maxpool(x)  # 4 x 4 x 16

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='cnn_128')
	return model	


def MobileNet_v2_035(inputs):
	x = inputs
	net = mobilenet_v2.mobilenet_v2_035	
	x, end_points = net(x, num_classes=5, is_training=True)
	#x = conv(x, f=8, k=3, s=1, p='SAME')
	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs, x, name='MobileNet_v2')	
	return model


def model_ResNet50(inputs):

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', 
		input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid', name=OUTPUT_NAME)(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_ResNet50')	
	return model