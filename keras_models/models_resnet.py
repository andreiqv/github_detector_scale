import tensorflow as tf
from tensorflow import keras
layers = keras.layers

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
#from tensorflow.keras.layers.merge import add
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

#----------

def conv(x, f, k, s=1, p='SAME'):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding='SAME',
		activation='tanh', # relu, selu
		use_bias=True)(x)
	return x

maxpool = lambda x, p=2: layers.MaxPool2D(pool_size=p, strides=1)(x)
	
bn = lambda x: layers.BatchNormalization()(x)


#-----------

def block(n_output, upscale=False):
	# n_output: number of feature maps in the block
	# upscale: should we use the 1x1 conv2d mapping for shortcut or not
	
	# keras functional api: return the function of type
	# Tensor -> Tensor
	def f(x):
		
		# H_l(x):
		# first pre-activation
		h = BatchNormalization()(x)
		h = Activation(relu)(h)
		# first convolution
		h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
		
		# second pre-activation
		h = BatchNormalization()(x)
		h = Activation(relu)(h)
		# second convolution
		h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
		
		# f(x):
		if upscale:
			# 1x1 conv2d
			f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
		else:
			# identity
			f = x
		
		# F_l(x) = f(x) + H_l(x):
		#return add([f, h])
		return f + h
	
	return f

def resnet18(inputs):

	x = inputs

	"""
	# input tensor is the 28x28 grayscale image
	#input_tensor = Input((28, 28, 1))

	input_tensor = inputs
	# first conv2d with post-activation to transform the input data to some reasonable form
	x = Conv2D(kernel_size=3, filters=16, strides=1, padding='SAME', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)

	# F_1
	x = block(16)(x)
	# F_2
	x = block(16)(x)

	# F_3
	# H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
	# and we can't add together tensors of inconsistent sizes, so we use upscale=True
	# x = block(32, upscale=True)(x)	   # !!! <------- Uncomment for local evaluation
	# F_4
	# x = block(32)(x)					 # !!! <------- Uncomment for local evaluation
	# F_5
	# x = block(32)(x)					 # !!! <------- Uncomment for local evaluation

	# F_6
	# x = block(48, upscale=True)(x)	   # !!! <------- Uncomment for local evaluation
	# F_7
	# x = block(48)(x)					 # !!! <------- Uncomment for local evaluation

	# last activation of the entire network's output
	x = BatchNormalization()(x)
	x = Activation(relu)(x)

	# average pooling across the channels
	# 28x28x48 -> 1x48
	x = GlobalAveragePooling2D()(x)

	# dropout for more robust learning
	x = Dropout(0.2)(x)

	# last softmax layer
	#x = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)
	
	#x = Activation(softmax)(x)
	#model = Model(inputs=input_tensor, outputs=x)
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])	

	"""


	#x = Dense(5, activation='sigmoid')(x)
	#x = layers.Dense(5, activation=None)(x)
	#model = Model(inputs, x, name='resnet18')	
	
	x = conv(x, f=8, k=3, s=1, p='VALID')
	x = maxpool(x)  # 64
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='SAME')
	x = maxpool(x)
	x = bn(x)
	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x)
	x = bn(x)
	x = conv(x, f=32, k=3, s=2, p='VALID')
	x = maxpool(x)
	x = bn(x)
	x = conv(x, f=32, k=3, s=1, p='VALID')
	x = maxpool(x)
	x = bn(x)	
	x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1000, activation='sigmoid')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	#x = layers.Dense(5, activation=None)(x)
	model = keras.Model(inputs, x, name='model_first_3')

	return model