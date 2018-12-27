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
	print(model.summary())
	session = keras.backend.get_session()

	print("model inputs:")
	for node in model.inputs:
		print(node.op.name)

	print("model outputs:")
	for node in model.outputs:
		print(node.op.name)

	freeze_graph(session.graph, session, \
		[out.op.name for out in model.outputs], path)



if __name__ == '__main__':

	model_name = 'model_3_2-165-1.000-1.000[0.818]'
	#model_name = 'model_first_3-60-1.000-1.000[0.803]'
	model_name = model_name.rstrip('.hdf5')

	base_model = keras.models.load_model(
		"./checkpoints/{}.hdf5".format(model_name),
		custom_objects={'miou': miou, 'accuracy': accuracy, 'bboxes_loss': bboxes_loss})
	
	modelsaver.save(model, path='../pb/' + model_name + '.pb')
