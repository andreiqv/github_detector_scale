import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from tensorflow.keras import regularizers



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



