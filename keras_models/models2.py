import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet	

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

def model_resnet50(inputs):

	net = resnet_v2.resnet_v2_50
	x, end_points = net(inputs, num_classes=5, is_training=True)
	x = layers.Reshape((5,))(x)
	model = keras.Model(inputs, x, name='resnet')
	return model

def model_resnet(inputs):

	base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', 
        input_tensor=inputs)
	x = base_model.output
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs=inputs, outputs=x, name='keras_model')	
	return model

	