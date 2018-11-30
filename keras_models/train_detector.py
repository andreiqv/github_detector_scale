import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

sys.path.append('.')
sys.path.append('..')
import keras_models.models as models
from tfrecords_converter import TfrecordsDataset
from keras_models.aux import miou, bboxes_loss, accuracy

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
image_shape = (128, 128)
image_channels = 3

K = keras.backend


# tfe = tf.contrib.eager
# tf.enable_eager_execution()

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def lr_scheduler(epoch, lr):
    decay_rate = 2 / 3.0
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


#dataset = TfrecordsDataset("../dataset/train-full128x128.tfrecords", "../dataset/test-full128x128.tfrecords", image_shape,
#                           image_channels, 256)

dataset = TfrecordsDataset("../dataset/train-bboxes128x128.tfrecords", 
                            "../dataset/test-bboxes128x128.tfrecords", 
                            image_shape, image_channels, 256)


dataset.augment_train_dataset()

inputs = keras.layers.Input(shape=(128, 128, 3))
#model = models.model_first2(inputs)
#model = models.model3(inputs)
#model = models.model_first(inputs)

import models2
model = models2.model_ResNet50(inputs)


# optimizer = tf.train.AdamOptimizer()
# optimizer = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              #optimizer='adagrad',
              #optimizer='adam',
              loss=bboxes_loss,
              metrics=[accuracy, miou])

# model = keras.models.load_model(
#     "./checkpoints/model2-106-0.991-0.991[0.645].hdf5",
#     custom_objects={'miou': miou, 'accuracy': accuracy, 'bboxes_loss': bboxes_loss}
# )

# model.compile(optimizer=optimizer,
#               loss=bboxes_loss,
#               metrics=[accuracy, miou])

print(model.summary())

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/model_test-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}[{val_miou:.3f}].hdf5",
        save_best_only=True,
        monitor='val_miou',
        mode='max'
    ),
    LRTensorBoard(
        log_dir='./tensorboard/model_test'
    ),
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]
keras.backend.get_session().run(tf.local_variables_initializer())

batch_size = 32  # 256

model.fit(dataset.train_set.batch(batch_size).repeat(),
          #callbacks=callbacks,
          #epochs=150,
          epochs=500,
          steps_per_epoch=123,
          validation_data=dataset.test_set.batch(batch_size).repeat(),
          validation_steps=6,
          )
