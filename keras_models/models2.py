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
	x = layers.Dense(5, activation='sigmoid')(x)
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


# --------------------

def conv(x, f, k, s=1):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding='SAME',
		activation='sigmoid', # relu
		use_bias=True)(x)
	return x

maxpool = lambda x, p=2: layers.MaxPool2D(pool_size=p, strides=1)(x)
	
bn = lambda x: layers.BatchNormalization()(x)

def model_cnn_128(inputs):
	""" val_accuracy: 0.9879 - val_miou: 0.6895  | val_miou: 0.7115
	with batchnorm: val_miou: 0.0750
	"""	
	x = inputs 
	x = conv(x, 8, 4)
	x = conv(x, 8, 4)
	x = maxpool(x)  # 64
	x = conv(x, 16, 4)
	x = conv(x, 16, 4)
	x = maxpool(x)  # 32
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 16
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 8
	x = conv(x, 32, 3)
	x = conv(x, 32, 3)
	x = maxpool(x)  # 4 x 4 x 16

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(1000, activation='elu')(x)
	#x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='cnn_128')
	return model	




def model_cnn_224(inputs):
	x = inputs 
	x = conv(x, 8, 3)
	x = conv(x, 8, 3)
	x = maxpool(x)  # 112
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 56
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 28
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 14
	x = conv(x, 16, 3)
	x = conv(x, 16, 3)
	x = maxpool(x)  # 7 x 7 x 16

	x = layers.Flatten()(x)
	#x = layers.Dropout(0.5)(x)
	#x = layers.Dense(1000, activation='elu')(x)
	#x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='cnn_224')
	
	return model	

"""
model_cnn_224

0/1: Epoch 9/500 - loss: 0.3312 - accuracy: 0.9556 - miou: 0.6948 
- val_loss: 0.3902 - val_accuracy: 0.9560 - val_miou: 0.6924

1: Epoch 6/500 - 454s 379ms/step - loss: 0.0011 - accuracy: 0.9978 - miou: 0.8414 
- val_loss: 0.0060 - val_accuracy: 0.9980 - val_miou: 0.7043

"""