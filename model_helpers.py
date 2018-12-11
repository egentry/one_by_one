import numpy as np
import os

import keras
from keras.applications import inception_v3, inception_resnet_v2, vgg19

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam


def build_model(input_shape, with_one_by_one=True,
                keep_training_pretrained=False,
                pretrained_class=vgg19.VGG19,
                num_dense=64,
                ):

    model = Sequential()

    # # # 1x1 convolution to make sure we only have 3 channels
    n_channels_for_pretrained = 3
    one_by_one = Conv2D(n_channels_for_pretrained, 1,
                        padding="same",
                        input_shape=input_shape)
    one_by_one.trainable = with_one_by_one
    model.add(one_by_one)

    pretrained_input_shape = tuple([n_channels_for_pretrained,
                                    *input_shape[1:]])
    pretrained_layers = pretrained_class(
        include_top=False,
        input_shape=pretrained_input_shape
    )

    for layer in pretrained_layers.layers:
        layer.trainable = keep_training_pretrained
    model.add(pretrained_layers)

    model.add(Flatten())
    model.add(Dense(2*num_dense, activation=None))
    model.add(keras.layers.PReLU())
    model.add(Dense(num_dense, activation=None))
    model.add(keras.layers.PReLU())
    model.add(Dense(1, activation=None))

    return model


def compile_model(model, logger_filename,
                  adam_lr=0.001,
                  ):

    learning_rate = adam_lr

    adam = Adam(lr=learning_rate)

    model.compile(loss="mean_squared_error",
                  optimizer=adam,
                  )

    # can only manually set weights _after_ compiling
    one_by_one_weights = np.zeros((1, 1, 5, 3))
    for i in range(3):
        # by default, irg should be RGB
        one_by_one_weights[0, 0, 3-i, i] = 1.
    model.layers[0].set_weights(
        [one_by_one_weights,
         np.zeros(3)])

    if os.path.exists(logger_filename):
        logger_filename_tmp = logger_filename + ".old"
        os.rename(logger_filename, logger_filename_tmp)

    return model
