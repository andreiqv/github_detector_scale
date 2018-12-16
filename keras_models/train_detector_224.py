"""
SqueezeNet
Darknet Reference  
Tiny Darknet
"""
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

if len(sys.argv) > 1 and sys.argv[1] == '1':
    presence = True
else:
    presence = False

if presence:
    from keras_models.aux1 import miou, bboxes_loss, accuracy
else:
    from keras_models.aux import miou, bboxes_loss, accuracy


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#image_shape = (128, 128)
image_shape = (224, 224)
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



inputs = keras.layers.Input(shape=(
    image_shape[0], image_shape[1], image_channels))

#model = models.model1(inputs) 
#model = models.model2(inputs) 
model = models.model3(inputs) 
#model = models.model4(inputs) 
#model = models.model_first(inputs)  
#model = models.model_first2(inputs)  # val_miou: 0.7990
#model = models.model_first_3(inputs)

import models2
#model = models2.model_cnn_224(inputs)

#model = models2.model_InceptionV3(inputs)
#model = models2.model_ResNet50(inputs)   #  0.7738 -> 0.8062
#model = models2.model_MobileNetV2(inputs)

#import new_keras_models.keras_darknet19 as keras_darknet19
#model = keras_darknet19.darknet19(inputs) # val_miou: 0.6106




# optimizer = tf.train.AdamOptimizer()
# optimizer = keras.optimizers.Adam(lr=0.0001)

print(model.summary())
num_layers = len(model.layers)
print('num_layers:', num_layers)
print('model.trainable_weights:', len(model.trainable_weights))

#num_last_trainable_layers = 60
#for layer in model.layers[:num_layers - num_last_trainable_layers]:
#   layer.trainable = False
#print('model.trainable_weights:', len(model.trainable_weights))


model.compile(optimizer=keras.optimizers.Adam(lr=0.0002),
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


# ------

batch_size = 16  # 128  (set 32 if size==224)

#dataset = TfrecordsDataset("../dataset/train-full128x128.tfrecords", "../dataset/test-full128x128.tfrecords", image_shape,
#                           image_channels, 256)

if presence:
    dataset = TfrecordsDataset("../dataset/presence_train-bboxes{}x{}.tfrecords".format(*image_shape), 
                            "../dataset/presence_test-bboxes{}x{}.tfrecords".format(*image_shape), 
                            image_shape, image_channels, batch_size)
    print('Using presence_train-bboxes{}x{}.tfrecords'.format(*image_shape))
    train_steps = 299 * 128 // batch_size # no pictures with empty scales
    valid_steps = 16  * 128 // batch_size # no pictures with empty scales

else:
    dataset = TfrecordsDataset("../dataset/train-bboxes{}x{}.tfrecords".format(*image_shape), 
                            "../dataset/test-bboxes{}x{}.tfrecords".format(*image_shape), 
                            image_shape, image_channels, batch_size)
    print('Using train-bboxes{}x{}.tfrecords'.format(*image_shape))
    train_steps = 469 * 128 // batch_size
    valid_steps = 24  * 128 // batch_size 

dataset.augment_train_dataset()

# ------

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

callbacksSave = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/model_test-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}[{val_miou:.3f}].hdf5",
        save_best_only=True,
        monitor='val_miou',
        mode='max'
    )
]

callbacksLearningRate = [   
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]



keras.backend.get_session().run(tf.local_variables_initializer())

model.fit(dataset.train_set.repeat(),
          #callbacks=callbacksLearningRate,
          #epochs=150,
          epochs=500,
          steps_per_epoch=train_steps,
          validation_data=dataset.test_set.batch(batch_size).repeat(),
          validation_steps=valid_steps,
          )
