# https://stackoverflow.com/questions/51622411/cant-import-frozen-graph-after-adding-layers-to-keras-model/51644241
# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer
# https://github.com/keras-team/keras/issues/11032
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io
import sys
sys.path.append('.')
sys.path.append('..')
from keras_models.aux import accuracy, bboxes_loss, miou
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


model_name = 'model_first2-213-0.982-0.982[0.778].hdf5'


def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, "./output", "{}.pb".format(model_name),
                             as_text=False)


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


keras.backend.set_learning_phase(0)

base_model = keras.models.load_model(
    "./checkpoints/{}.hdf5".format(model_name),
    custom_objects={'miou': miou, 'accuracy': accuracy, 'bboxes_loss': bboxes_loss}
)

# input_tensor = keras.layers.Input(shape=(128, 128, 3), batch_size=1)
# base_model.layers.pop(0)
# new_outputs = base_model(input_tensor)
# base_model = keras.Model(inputs=input_tensor, outputs=new_outputs)

print(base_model.summary())

session = keras.backend.get_session()

print("model inputs")
for node in base_model.inputs:
    print(node.op.name)

print("model outputs")
for node in base_model.outputs:
    print(node.op.name)

freeze_graph(session.graph, session, [out.op.name for out in base_model.outputs])
