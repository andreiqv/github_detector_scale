import tensorflow as tf
from tensorflow import keras

layers = keras.layers

def model1(inputs):
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='elu',
        use_bias=True)(inputs)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation=None
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        filters=5,
        kernel_size=(8, 8),
        strides=(1, 1),
        padding='VALID',
        use_bias=True,
        activation='sigmoid'
    )(x)

    x = layers.Reshape((5,))(x)
    model = keras.Model(inputs, x, name='glp_model1')

    return model


def model2(inputs):
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='elu',
        use_bias=True)(inputs)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation=None
    )(x)
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='elu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=2
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        filters=5,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='VALID',
        use_bias=True,
        activation='sigmoid'
    )(x)

    x = layers.Reshape((5,))(x)
    model = keras.Model(inputs, x, name='glp_model1')

    return model


def darknet_block(filters, kernel, stride, padding, x):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        padding=padding,
        activation=None,
        use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    return x


def model3(inputs):
    x = darknet_block(32, 3, 1, 'SAME', inputs)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(64, 3, 1, 'SAME',  x)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(128, 3, 1, 'SAME', x)
    x = darknet_block(64, 1, 1, 'SAME',  x)
    x = darknet_block(128, 3, 1, 'SAME', x)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(256, 3, 1, 'SAME', x)
    x = darknet_block(64, 3, 2, 'VALID', x)

    x = layers.Conv2D(
        filters=5,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='VALID',
        use_bias=False,
        activation='sigmoid'
    )(x)

    x = layers.Reshape((5,))(x)
    model = keras.Model(inputs, x, name='glp_model3')

    return model


def model4(inputs):  # added
    x = darknet_block(32, 3, 1, 'SAME', inputs)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(64, 3, 1, 'SAME',  x)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(128, 3, 1, 'SAME', x)
    x = darknet_block(64, 1, 1, 'SAME',  x)
    x = darknet_block(128, 3, 1, 'SAME', x)
    x = darknet_block(32, 3, 2, 'VALID', x)
    x = darknet_block(256, 3, 1, 'SAME', x)
    x = darknet_block(64, 3, 2, 'VALID', x)
    
    """
    x = layers.Conv2D(
        filters=5,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='VALID',
        use_bias=False,
        activation='sigmoid'
    )(x)
    """

    #x = layers.Reshape((5,))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, x, name='glp_model3')

    return model

#------

def model_first(inputs):
    x = layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        activation='relu',
        use_bias=True)(inputs)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='relu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='relu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='relu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        use_bias=True,
        activation='relu'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, x, name='first_glp_model')

    return model


def model_first2(inputs):
    x = layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        activation='tanh',
        use_bias=True)(inputs)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation='sigmoid')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='sigmoid')(x)
    #x = layers.Dense(5, activation=None)(x)

    model = keras.Model(inputs, x, name='model_first2')

    return model


def model_first_3(inputs):
    x = layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        activation='tanh',
        use_bias=True)(inputs)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='VALID',
        use_bias=True,
        activation='tanh'
    )(x)
    x = layers.MaxPool2D(
        pool_size=2,
        strides=1
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='sigmoid')(x)
    #x = layers.Dense(5, activation=None)(x)

    model = keras.Model(inputs, x, name='model_first_3')

    return model
