import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet	

def model_resnet(inputs):

	net = resnet_v2.resnet_v2_50
	x, end_points = net(inputs, num_classes=5, is_training=True)
	x = layers.Reshape((5,))(x)
	model = keras.Model(inputs, x, name='glp_model1')
	return model