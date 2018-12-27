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


def MobileNet_v2_035(inputs):

	net = mobilenet_v2.mobilenet_v2_035	
	x, end_points = net(inputs, num_classes=5, is_training=True)
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