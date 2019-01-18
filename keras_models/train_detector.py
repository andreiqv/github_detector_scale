"""
python3 keras_models/train_detector.py 1

"1" means that presence_train-bboxes will be used (without empty scales)
"""
import os
import sys
#import cv2
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
    # localization
    from keras_models.aux import bboxes_loss
    learning_rate = 0.0005
else:
    # objectness
    from keras_models.aux import bboxes_loss_objectness as bboxes_loss
    learning_rate = 0.01

from keras_models.aux import miou, accuracy

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
image_shape = (128, 128)
#image_shape = (224, 224)
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
    image_shape[0], image_shape[1], image_channels), name='input')


#model = models.model1(inputs)  # val_miou: 0.0517 -> 0.0855
#model = models.model2(inputs)  # val_miou: 0.6534 -> 0.7436
#model = models.model3(inputs)  # val_miou: 0.7457 -> 0.7912
#model = models.model4(inputs)  # val_miou:  0.7663 -> 0.7925
#model = models.model_first(inputs)  # val_miou: 0.7519 ->  0.7715
#model = models.model_first2(inputs) # val_miou: 0.7731  -> 0.8045 
#model = models.model_first2_1(inputs)

import models2
#model = models2.model_3(inputs)   # val_miou: 0.8053
#model = models2.model_3_1(inputs) # val_miou: 0.8113 ++
#model = models2.model_3_2(inputs) # val_miou: 0.8118 
#model = models2.model_3_3(inputs) # val_miou: 0.8049
#model = models2.model_cnn_128(inputs)
#model = models2.model_cnn_128_v2(inputs)

#model = models2.model_InceptionV3(inputs)              # val_miou: 0.8241
#model = models2.model_ResNet50(inputs)                 # val_miou: 0.8524
#model = models2.model_MobileNet(inputs, depth=1)       # val_miou: 0.8500
#model = models2.model_MobileNetV2(inputs, depth=0.35)  # val_miou: 0.8022
model = models2.model_MobileNetV2(inputs, depth=1)      # val_miou: 0.8460

#import models_slim
#model = models_slim.model_cnn_128(inputs)     
#model = models_slim.MobileNet_v2_035(inputs)     


#import new_keras_models.keras_darknet19 as keras_darknet19
#model = keras_darknet19.darknet19(inputs) # val_miou: 0.6106

#import models_resnet
#model = models_resnet.resnet18(inputs)

#import models3
#model = models3.resnet_keras(inputs)

#import resnet_v2
#model = resnet_v2.ResnetBuilder.build_resnet_18(  # val_miou: 0.8286
#               (image_channels, image_shape[0], image_shape[1]), 5)

model_name = 'model_MobileNetV2'
model_name = 'presence_' + model_name if presence else model_name


# optimizer = tf.train.AdamOptimizer()
# optimizer = keras.optimizers.Adam(lr=0.0001)

print(model.summary())
num_layers = len(model.layers)
print(model_name)
print('num_layers:', num_layers)
print('model.trainable_weights:', len(model.trainable_weights))


#num_last_trainable_layers = 60
#for layer in model.layers[:num_layers - num_last_trainable_layers]:
#   layer.trainable = False
#print('model.trainable_weights:', len(model.trainable_weights))


model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
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

batch_size = 64  # 128  (set 32 if size==224)

#dataset = TfrecordsDataset("../dataset/train-full128x128.tfrecords", "../dataset/test-full128x128.tfrecords", image_shape,
#                           image_channels, 256)

if presence:
    train_path = "../dataset/bg-presence-train-bboxes{}x{}.tfrecords".format(*image_shape)
    test_path = "../dataset/bg-presence-test-bboxes{}x{}.tfrecords".format(*image_shape)
    print('Using presence dataset {}x{}'.format(*image_shape))
    train_steps = 586 * 128 // batch_size # no pictures with empty scales
    valid_steps = 16  * 128 // batch_size # no pictures with empty scales

else:
    train_path = "../dataset/bg-train-bboxes{}x{}.tfrecords".format(*image_shape)
    test_path = "../dataset/bg-test-bboxes{}x{}.tfrecords".format(*image_shape)
    print('Using full dataset {}x{}'.format(*image_shape))
    train_steps = 788 * 128 // batch_size
    valid_steps = 24  * 128 // batch_size 


print('train dataset path:', train_path)
print('test  dataset path:', test_path)
dataset = TfrecordsDataset(train_path, test_path, image_shape, image_channels, batch_size)
dataset.augment_train_dataset()


# ------

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/"+model_name+"-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}[{val_miou:.3f}].hdf5",
        save_best_only=True,
        monitor='val_miou',
        mode='max'
    ),
    LRTensorBoard(
        log_dir='./tensorboard/{}'.format(model_name)
    ),
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]

callbacksSave = [   
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/"+model_name+"-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}[{val_miou:.3f}].hdf5",
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
          #callbacks=callbacksSave,
          callbacks=callbacks,
          epochs=1000, 
          #epochs=1000,
          steps_per_epoch=train_steps,
          validation_data=dataset.test_set.batch(batch_size).repeat(),
          validation_steps=valid_steps,
          )

# save into PB
keras.backend.set_learning_phase(0)

model.save_weights("./checkpoints/saved_{}.hdf5".format(model_name))

import modelsaver
modelsaver.save(model, path='../pb/', filename=model_name+'.pb')
