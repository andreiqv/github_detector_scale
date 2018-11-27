import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tfrecords_converter import TfrecordsDataset

EMPTY_PLATFORM_THRESHOLD = 0.7
image_shape = (128, 128)
image_channels = 3

K = keras.backend


# tfe = tf.contrib.eager
# tf.enable_eager_execution()


def bboxes_loss(labels, logits):
    objectness_loss = tf.contrib.losses.mean_squared_error(logits[:, 4], labels[:, 4])
    bbox_loss = tf.reduce_mean(tf.squared_difference(logits[:, :4], labels[:, :4]), axis=1)
    bbox_loss = tf.reduce_mean(bbox_loss * labels[:, 4])
    return objectness_loss + bbox_loss


def accuracy(_labels, _logits):
    def overlap(logits, labels, center, dimension):
        logits_w_2 = tf.divide(logits[:, dimension], tf.constant(2.0))
        labels_w_2 = tf.divide(labels[:, dimension], tf.constant(2.0))
        l1 = logits[:, center] - logits_w_2
        l2 = labels[:, center] - labels_w_2
        left = tf.maximum(l1, l2)

        r1 = logits[:, center] + logits_w_2
        r2 = labels[:, center] + labels_w_2

        right = tf.minimum(r1, r2)

        width = tf.subtract(right, left)

        # https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
        return tf.nn.relu(width)

    correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    threshold = tf.constant(EMPTY_PLATFORM_THRESHOLD)

    empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    not_empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty = tf.equal(_labels[:, 4], tf.constant([1.0]))
    empty = tf.logical_not(not_empty)

    logit_not_empty = tf.greater_equal(_logits[:, 4], threshold)
    logit_empty = tf.logical_not(logit_not_empty)

    correct_mask_e = tf.logical_and(empty, logit_empty)
    correct_mask_n = tf.logical_and(not_empty, logit_not_empty)
    wrong_mask_e = tf.logical_and(empty, logit_not_empty)
    wrong_mask_n = tf.logical_and(not_empty, logit_empty)

    empty_correct = empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32))
    empty_wrong = empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_e))[0], dtype=tf.float32))

    not_empty_correct = not_empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_n))[0], dtype=tf.float32))
    not_empty_wrong = not_empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_n))[0], dtype=tf.float32))

    correct = correct.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_not_empty)))[0] + \
                                         tf.shape(tf.where(tf.logical_and(empty, logit_empty)))[0], dtype=tf.float32))

    wrong = wrong.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_empty)))[0] + \
                                     tf.shape(tf.where(tf.logical_and(empty, logit_not_empty)))[0], dtype=tf.float32))

    w = overlap(_logits, _labels, 0, 2)
    h = overlap(_logits, _labels, 1, 3)
    intersection = tf.multiply(w, h)
    union = tf.multiply(_logits[:, 2], (_logits[:, 3])) + tf.multiply(_labels[:, 2], (_labels[:, 3])) - intersection
    iou = tf.divide(intersection, union)
    correct_empty_amount = tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32)
    mean_iou = tf.reduce_sum(tf.boolean_mask(iou, correct_mask_n))
    mean_iou = tf.divide(mean_iou, tf.cast(tf.shape(iou)[0], dtype=tf.float32) - correct_empty_amount)

    return correct / (correct + wrong)


def miou(_labels, _logits):
    def overlap(logits, labels, center, dimension):
        logits_w_2 = tf.divide(logits[:, dimension], tf.constant(2.0))
        labels_w_2 = tf.divide(labels[:, dimension], tf.constant(2.0))
        l1 = logits[:, center] - logits_w_2
        l2 = labels[:, center] - labels_w_2
        left = tf.maximum(l1, l2)

        r1 = logits[:, center] + logits_w_2
        r2 = labels[:, center] + labels_w_2

        right = tf.minimum(r1, r2)

        width = tf.subtract(right, left)

        # https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
        return tf.nn.relu(width)

    correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    threshold = tf.constant(EMPTY_PLATFORM_THRESHOLD)

    empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty_correct = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    not_empty_wrong = tf.Variable(0, dtype=tf.float32, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    not_empty = tf.equal(_labels[:, 4], tf.constant([1.0]))
    empty = tf.logical_not(not_empty)

    logit_not_empty = tf.greater_equal(_logits[:, 4], threshold)
    logit_empty = tf.logical_not(logit_not_empty)

    correct_mask_e = tf.logical_and(empty, logit_empty)
    correct_mask_n = tf.logical_and(not_empty, logit_not_empty)
    wrong_mask_e = tf.logical_and(empty, logit_not_empty)
    wrong_mask_n = tf.logical_and(not_empty, logit_empty)

    empty_correct = empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32))
    empty_wrong = empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_e))[0], dtype=tf.float32))

    not_empty_correct = not_empty_correct.assign_add(tf.cast(tf.shape(tf.where(correct_mask_n))[0], dtype=tf.float32))
    not_empty_wrong = not_empty_wrong.assign_add(tf.cast(tf.shape(tf.where(wrong_mask_n))[0], dtype=tf.float32))

    correct = correct.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_not_empty)))[0] + \
                                         tf.shape(tf.where(tf.logical_and(empty, logit_empty)))[0], dtype=tf.float32))

    wrong = wrong.assign_add(tf.cast(tf.shape(tf.where(tf.logical_and(not_empty, logit_empty)))[0] + \
                                     tf.shape(tf.where(tf.logical_and(empty, logit_not_empty)))[0], dtype=tf.float32))

    w = overlap(_logits, _labels, 0, 2)
    h = overlap(_logits, _labels, 1, 3)
    intersection = tf.multiply(w, h)
    union = tf.multiply(_logits[:, 2], (_logits[:, 3])) + tf.multiply(_labels[:, 2], (_labels[:, 3])) - intersection
    iou = tf.divide(intersection, union)
    correct_empty_amount = tf.cast(tf.shape(tf.where(correct_mask_e))[0], dtype=tf.float32)
    mean_iou = tf.reduce_sum(tf.boolean_mask(iou, correct_mask_n))
    mean_iou = tf.divide(mean_iou, tf.cast(tf.shape(iou)[0], dtype=tf.float32) - correct_empty_amount)

    return mean_iou


dataset = TfrecordsDataset("../dataset/train128x128.tfrecords", "../dataset/test128x128.tfrecords", image_shape,
                           image_channels,
                           100)

dataset.augment_train_dataset()

# inputs = keras.layers.Input(shape=(128, 128, 3))

base_model = MobileNetV2(input_shape=(128, 128, 3), alpha=0.35, include_top=False)

x = keras.layers.Flatten()(base_model.output)
predictions = keras.layers.Dense(5, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

for layer in base_model.layers[:-4]:
    layer.trainable = False

optimizer = tf.train.AdamOptimizer()

model.compile(optimizer='adam',
              loss=bboxes_loss,
              metrics=[accuracy, miou])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/mobilenetv2-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}[{val_miou:.3f}].hdf5",
        save_best_only=True,
        monitor='val_miou',
        mode='max'
    ),
    keras.callbacks.TensorBoard(
        log_dir='./tensorboard',
        write_images=True,
    )
]
keras.backend.get_session().run(tf.local_variables_initializer())

model.fit(dataset.train_set.repeat(),
          callbacks=callbacks,
          epochs=100,
          steps_per_epoch=290,
          validation_data=dataset.test_set.batch(32).repeat(),
          validation_steps=20,
          )
