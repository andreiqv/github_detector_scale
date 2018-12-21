import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.keras import regularizers



def conv(x, f, k, s=1, p='SAME'):
	x = layers.Conv2D(
		filters=f,
		kernel_size=(k, k),
		strides=(s, s),
		padding='SAME',
		activation='sigmoid', # relu, selu
		#kernel_regularizer=regularizers.l2(0.01),
		use_bias=True)(x)
	return x

maxpool = lambda x, p=2, s=1: layers.MaxPool2D(pool_size=p, strides=s)(x)
	
maxpool2 = lambda x, p=2: layers.MaxPool2D(pool_size=p)(x)

bn = lambda x: layers.BatchNormalization()(x)



def model_first_64(inputs):
	""" 
	2 conv layers: val_accuracy: 0.9333 - val_miou: 0.4136

	+bn: val_accuracy: ep=4 - val_accuracy: 0.9309 - val_miou: 0.5508

	val_accuracy: 0.9416 - val_miou: 0.5313

	Epoch 78/500 - 28s 119ms/step 
	- loss: 0.0190 - accuracy: 0.9769 - miou: 0.6496 
	- val_loss: 0.0774 - val_accuracy: 0.9769 - val_miou: 0.5466
	"""
	x = inputs
	x = conv(x, f=8, k=3, s=2, p='VALID')
	x = maxpool(x) # 32
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 8
	x = bn(x)
	
	x = conv(x, f=16, k=3, s=1, p='VALID')
	x = maxpool(x) # 4	
	x = bn(x)
	print('x shape:', x.get_shape()) # (?, 6, 6, 32)

	x = layers.Flatten()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(500, activation='sigmoid')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	#x = layers.Dense(5, activation=None)(x)
	model = keras.Model(inputs, x, name='model_first_3')
	return model	


def model_first_64_v2(inputs):
	""" 
	10: val_accuracy: 0.9659 - val_miou: 0.6528
	20: val_accuracy: 0.9703 - val_miou: 0.6390
	30: val_accuracy: 0.9752 - val_miou: 0.6339
	--
	30: val_accuracy: 0.9801 - val_miou: 0.6330
	96: val_accuracy: 0.9883 - val_miou: 0.6413
	"""
	x = inputs
	x = conv(x, f=8, k=3, s=2, p='VALID')
	x = maxpool(x) # 32
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 8
	x = bn(x)
	
	x = conv(x, f=16, k=3, s=1, p='VALID')
	x = maxpool(x) # 4	
	x = bn(x)
	print('x shape:', x.get_shape()) #

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='model_first_64')
	return model	


def model_first_64_v3(inputs):
	""" 
	Epoch 30:  val_accuracy: 0.9743 - val_miou: 0.6595
	Epoch 85:  val_accuracy: 0.9858 - val_miou: 0.6619
	Epoch 264/500 - val_loss: 0.0303 - val_accuracy: 0.9921 - val_miou: 0.6973
	Epoch 00459: LearningRateScheduler reducing learning rate to 1.3365716995394905e-06.
	Epoch 459/500 - 28s 119ms/step 
	- loss: 0.0063 - accuracy: 0.9934 - miou: 0.7458 
	- val_loss: 0.0306 - val_accuracy: 0.9934 - val_miou: 0.6960

	"""
	x = inputs
	x = conv(x, f=8, k=3, s=2, p='SAME')
	x = maxpool(x) # 32
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='SAME')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='SAME')
	x = maxpool(x) # 8
	x = bn(x)
	
	x = conv(x, f=16, k=3, s=1, p='SAME')
	x = maxpool(x) # 4	
	x = bn(x)
	print('x shape:', x.get_shape()) #

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='model_first_64')
	return model	



def model_first_64_v4(inputs):
	""" 
learning_rate = 0.01

10:  val_accuracy: 0.9624 - val_miou: 0.5250
20:  val_accuracy: 0.9699 - val_miou: 0.5778
50:  val_accuracy: 0.9810 - val_miou: 0.6587
100: val_accuracy: 0.9873 - val_miou: 0.6445
---	
без последних слоёв:
10:  val_accuracy: 0.9648 - val_miou: 0.5810 
20:  val_accuracy: 0.9707 - val_miou: 0.6323
50:  val_accuracy: 0.9807 - val_miou: 0.6427
100: val_accuracy: 0.9873 - val_miou: 0.6742
300: val_accuracy: 0.9929 - val_miou: 0.6910
---
2conv подряд:
10: val_accuracy: 0.9689 - val_miou: 0.4656
20: val_accuracy: 0.9729 - val_miou: 0.6325
50: val_accuracy: 0.9822 - val_miou: 0.6384

	"""
	x = inputs
	x = conv(x, f=8, k=3, s=2, p='VALID')
	x = conv(x, f=8, k=3, s=1, p='SAME')
	x = maxpool(x) # 32
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 8
	x = bn(x)
	
	#x = conv(x, f=16, k=3, s=2, p='VALID')	
	#x = maxpool(x) # 4
	#x = bn(x)
	print('x shape:', x.get_shape()) #

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='model_first_64')
	return model	


def model_first_64_v5(inputs):
	""" 
learning_rate = 0.01
10: val_accuracy: 0.9678 - val_miou: 0.6306
20: val_accuracy: 0.9735 - val_miou: 0.6088
30: val_accuracy: 0.9779 - val_miou: 0.5900

	"""
	x = inputs
	x = conv(x, f=8, k=3, s=2, p='VALID')
	x = bn(x)
	x = conv(x, f=8, k=3, s=1, p='SAME')
	x = maxpool(x) # 32
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')	
	x = maxpool(x) # 8
	x = bn(x)
	
	#---
	#x = conv(x, f=16, k=3, s=1, p='VALID')	 # add
	#x = maxpool(x) # 4	
	#x = bn(x)	# add
	print('x shape:', x.get_shape()) #

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='model_first_64')
	return model	


def model_first_64_v6(inputs):
	""" 
01: val_accuracy: 0.9278 - val_miou: 0.0707
05: val_accuracy: 0.9566 - val_miou: 0.5089
10: val_accuracy: 0.9657 - val_miou: 0.6342
	"""
	x = inputs
	x1 = conv(x, f=8, k=3, s=2, p='VALID')
	x1 = bn(x1)
	x2 = conv(x, f=8, k=3, s=1, p='SAME')
	x2 = maxpool2(x2) # 32
	x2 = bn(x2)
	x3 = maxpool2(x)
	x = layers.concatenate([x1, x2, x3])	
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')
	x = maxpool(x) # 16
	x = bn(x)

	x = conv(x, f=16, k=3, s=2, p='VALID')	
	x = maxpool(x) # 8
	x = bn(x)
	
	#---
	#x = conv(x, f=16, k=3, s=1, p='VALID')	 # add
	#x = maxpool(x) # 4	
	#x = bn(x)	# add
	print('x shape:', x.get_shape()) #

	x = layers.Flatten()(x)
	x = layers.Dense(5, activation='sigmoid')(x)
	model = keras.Model(inputs, x, name='model_first_64')
	return model	

