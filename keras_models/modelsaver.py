import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io
import sys
sys.path.append('.')
sys.path.append('..')
from keras_models.aux import accuracy, bboxes_loss, miou
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def freeze_graph(graph, session, output, path):
	with graph.as_default():
		graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
		graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
		graph_io.write_graph(graphdef_frozen, path, as_text=False)


def save(model, path):

	keras.backend.set_learning_phase(0)
	print(base_model.summary())
	session = keras.backend.get_session()

	print("model inputs:")
	for node in base_model.inputs:
		print(node.op.name)

	print("model outputs:")
	for node in base_model.outputs:
		print(node.op.name)

	freeze_graph(session.graph, session, \
		[out.op.name for out in base_model.outputs], path)