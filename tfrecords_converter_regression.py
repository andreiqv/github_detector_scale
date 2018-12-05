import tensorflow as tf
import math
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# tf.enable_eager_execution()

"""
'tf.train.Example': (
    'feature': 'tf.train.Features':(
        "image": 'tf.train.Feature': (float_list=
            'tf.train.FloatList(value=value)'),

        "label": 'tf.train.Feature': (bytes_list=
            'tf.train.BytesList(value=[value])'),

        "data": 'tf.train.Feature': (float_list=
            'tf.train.Int64List(value=value)'
        )
    )
)
"""
"""
'tf.train.SequenceExample': (
    context=customer,
    feature_lists= 'tf.train.FeatureLists':(
        feature_list={
            'Movie Names': 'tf.train.FeatureList': (
                feature=[
                    tf.train.Feature(bytes_list='tf.train.BytesList(value=[value]))
                ]
            )
        }
    )
)
"""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def plot_with_label(im_path, label):
    im = Image.open(im_path).resize(image_size, Image.BICUBIC)

    x1 = (label[0] - label[2] / 2.0) * im.size[0]
    x2 = (label[0] + label[2] / 2.0) * im.size[0]

    y1 = (label[1] - label[3] / 2.0) * im.size[1]
    y2 = (label[1] + label[3] / 2.0) * im.size[1]

    draw = ImageDraw.Draw(im)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))
    del draw
    plt.imshow(im)
    plt.show()


def convert_to_tfrecords(x_list, y_list, image_size, output_file):
    flat_images = []

    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for i, image_path in tqdm(enumerate(x_list), total=len(x_list)):
            label_path = y_list[i]

            label = None
            objectness = 1
            with open(label_path, "r") as l_f:
                for line in l_f:
                    label = list(map(float, line.strip().split(" ")))
                    break
            if label[0] == label[1] == label[2] == label[3] == 0:
                objectness = 0
            #label.append(objectness)
            #label = [0, 1] if objectness == 0 else [1, 0]
            label = [objectness]

            im = Image.open(image_path).resize(image_size, Image.BICUBIC)
            np_im = np.array(im)
            flat_image = np.reshape(np_im, (np_im.size,))
            flat_images.append(flat_image)

            features = tf.train.Features(
                feature={
                    "image": _int64_feature(flat_image.tolist()),
                    "label": _float_feature(label)
                }
            )
            example = tf.train.Example(features=features)

            # Write images with a good only
            record_writer.write(example.SerializeToString())


def plot_image(image):
    plt.imshow(image)
    plt.show()


class TfrecordsDataset:

    def __init__(self, train_file, test_file, image_size, channels_count, batch_size) -> None:
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.train_set = self.load_tfrecords_file(self.train_file, image_size, channels_count)
        self.test_set = self.load_tfrecords_file(self.test_file, image_size, channels_count)

    def load_tfrecords_file(self, file_name, image_size, channels_count):
        def _convert_tfrecord_to_tensor(example_proto):
            features = {
                'image': tf.VarLenFeature(dtype=tf.int64),
                'label': tf.VarLenFeature(dtype=tf.float32),
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            image = parsed_features["image"].values

            image = tf.cast(image, tf.float32)
            # image = tf.image.per_image_standardization(image)

            image = tf.divide(image, tf.constant([255.0], dtype=tf.float32))

            image = tf.reshape(image, (image_size[1], image_size[0], channels_count))

            # image = tf.reshape(image, (image_size[1], image_size[0], channels_count))
            # image = tf.image.per_image_standardization(image)
            # image = tf.expand_dims(image, 0)

            label_tensor = tf.convert_to_tensor(parsed_features["label"].values, dtype=tf.float32)

            return image, label_tensor

        dataset = tf.data.TFRecordDataset(file_name)

        dataset = dataset.map(_convert_tfrecord_to_tensor)
        return dataset

    def augment_train_dataset(self):
        self.train_set = self.train_set.shuffle(60000).repeat(5).batch(self.batch_size)

        def _random_distord(images, labels):
            rand = tf.random_uniform(shape=(1,), minval=0, maxval=2)
            toss = tf.less(rand, tf.constant([1.0], dtype=tf.float32))
            toss = tf.reshape(toss, [])

            def _flip_left():
                fliped_im = tf.image.flip_left_right(images)

                mirror = tf.constant([-1], dtype=tf.float32)
                mirror = tf.expand_dims(mirror, 0)
                mirror = tf.pad(mirror, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT", constant_values=1)
                mirror = tf.squeeze(mirror)
                mirror = tf.multiply(labels, mirror)  # [-0.34, 0.4, 0.28, 0,21, 1]

                basis = tf.constant([1], dtype=tf.float32)
                basis = tf.expand_dims(basis, 0)
                basis = tf.pad(basis, [[0, 0], [0, tf.shape(labels)[1] - 1]], "CONSTANT")  # [1, 0, 0 0 0]
                basis = tf.squeeze(basis)  # ?

                fliped_label = basis + mirror  # [0.6599, 0.4, 0.28, 0.21, 1]

                # eliminate fucking empty image coordinates problem
                # https://stackoverflow.com/questions/50538038/tf-data-dataset-mapmap-func-with-eager-mode
                new_labels = tf.multiply(fliped_label[:, 0], fliped_label[:, 4])
                new_labels = tf.expand_dims(new_labels, 1)
                new_labels = tf.concat([new_labels, fliped_label[:, 1:]], 1)

                return fliped_im, new_labels

            images, labels = tf.cond(toss, _flip_left, lambda: (images, labels))

            toss2 = tf.less(tf.random_uniform(shape=(1,), minval=0, maxval=4), tf.constant([1.0], dtype=tf.float32))
            toss2 = tf.reshape(toss2, [])

            def _random_gray_scale():
                gray_ims = tf.image.rgb_to_grayscale(images)
                rgb_gray_ims = tf.image.grayscale_to_rgb(gray_ims)
                return rgb_gray_ims, labels

            images, labels = tf.cond(toss2, _random_gray_scale, lambda: (images, labels))

            images = tf.image.random_hue(images, max_delta=0.05)
            images = tf.image.random_contrast(images, lower=0.7, upper=1.5)
            images = tf.image.random_brightness(images, max_delta=0.1)
            images = tf.image.random_saturation(images, lower=1.0, upper=1.5)

            images = tf.minimum(images, 1.0)
            images = tf.maximum(images, 0.0)

            return images, labels

        self.train_set = self.train_set.map(_random_distord, num_parallel_calls=2).prefetch(1)

        return self


if __name__ == '__main__':
    test_percent = 0.2
    #image_size = (128, 96)
    image_size = (128, 128)
    channels_count = 3

    dataset_list = 'dataset-bboxes.list'

    files_list = []
    labels_list = []
    with open(dataset_list, "r") as d_l:
        for line in d_l:
            files_list.append(line.strip())
            labels_list.append(line.replace(".jpg", ".txt").strip())
    test_split_index = math.ceil(len(files_list) * test_percent)

    files_list = np.array(files_list)
    labels_list = np.array(labels_list)

    idx = np.arange(len(files_list))
    np.random.shuffle(idx)
    test_split = idx[:test_split_index]
    train_split = idx[test_split_index:]

    x_test = files_list[test_split]
    y_test = labels_list[test_split]

    x_train = files_list[train_split]
    y_train = labels_list[train_split]

    # dataset = read_tf_records("training.tfrecords", image_size, channels_count)

    print(len(x_train))
    print(len(x_test))
    convert_to_tfrecords(x_train, y_train, image_size, 
                "../dataset/regression_train-bboxes{}x{}.tfrecords".format(image_size[1], image_size[0]))
    convert_to_tfrecords(x_test, y_test, image_size,
                "../dataset/regression_test-bboxes{}x{}.tfrecords".format(image_size[1], image_size[0]))
