import tensorflow as tf
slim = tf.contrib.slim
from settings import IMAGE_SIZE

from resnet_18 import resnet

def resnet18(inputs, num_classes=1000, is_training=True):

	hp = resnet.HParams(batch_size=32,
		num_gpus=1,
		num_classes=148,
		weight_decay=0.0002,
		momentum=0.001,
		finetune=False)

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0]*IMAGE_SIZE[1]*3])

	network_train = resnet.ResNet(hp, x train_labels, global_step, name="train")
	network_train.build_model()
	network_train.build_train_op()

	logits = slim.fully_connected(x, num_classes, scope='fc/fc_1')
	end_points = ['none']
	return logits, end_points

