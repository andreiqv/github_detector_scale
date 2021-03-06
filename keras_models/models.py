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
        activation='tanh',   # 0.7757 -> tanh 0.8070
        use_bias=False)(x)
    #x = layers.LeakyReLU(0.1)(x)
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
    x = darknet_block(256, 3, 1, 'SAME', x) # +

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
    """
    Epoch 31/500 - loss: 0.0023 - accuracy: 0.9846 - miou: 0.7896 
    - val_loss: 0.0032 - val_accuracy: 0.9848 - val_miou: 0.7774

    Epoch 105/500 - 112s 187ms/step 
    - loss: 6.4420e-04 - accuracy: 0.9993 - miou: 0.8704 
    - val_loss: 0.0033 - val_accuracy: 0.9994 - val_miou: 0.7837
    """
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
    """ Epoch 69/500 - 65s 108ms/step 
    - loss: 0.0028 - accuracy: 1.0000 - miou: 0.7747 
    - val_loss: 0.0033 - val_accuracy: 1.0000 - val_miou: 0.7833

    """
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


def model_first2_1(inputs):
    """
    Epoch 147/500 - loss: 0.0011 - accuracy: 0.9999 - miou: 0.8397 
    - val_loss: 0.0029 - val_accuracy: 0.9999 - val_miou: 0.8023

    224x224:
    Epoch 232/500- 199s 166ms/step - loss: 6.0059e-04 - accuracy: 1.0000 
    - miou: 0.8748 - val_loss: 0.0030 - val_accuracy: 1.0000 - val_miou: 0.7983

    1) 0-1
    learning_rate = 0.001
    Epoch 382/500: val_accuracy: 0.9944 - val_miou: 0.7426
    """
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

    # add
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        use_bias=True,
        activation='tanh'
    )(x)
    # ---

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

    model = keras.Model(inputs, x, name='model_first2_1')

    return model




