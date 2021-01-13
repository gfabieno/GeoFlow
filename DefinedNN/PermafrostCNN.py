# -*- coding: utf-8 -*-
"""Build and train a CNN on permafrost models."""

from tensorflow.keras.layers import (Conv2D, MaxPool2D, Flatten, Dense,
                                     Reshape, LeakyReLU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

from GeoFlow.NN import Hyperparameters, NN
from GeoFlow.Losses import mean_squared_error


class Hyperparameters(Hyperparameters):
    def __init__(self):
        self.restore_from = None
        self.epochs = 5
        self.steps_per_epoch = 100
        self.batch_size = 10

        # The learning rate.
        self.learning_rate = 1E-3
        # Adam optimizer hyperparameters.
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-5


class PermafrostCNN(NN):
    toinputs = ["dispersion"]
    tooutputs = ["vpdepth"]

    def build_network(self, inputs: dict):
        self.conv_blocks = [build_conv_block([16, 32]),
                            build_conv_block([64, 128]),
                            build_conv_block([128, 64])]
        self.flatten = Flatten()
        self.dense = Dense(400, activation=softmax)
        self.reshape = Reshape(target_shape=(400, 1))

    def call(self, inputs: dict):
        data_stream = inputs['dispersion']
        data_stream = data_stream[..., 0, :]
        for conv_block in self.conv_blocks:
            for layer in conv_block:
                data_stream = layer(data_stream)
        data_stream = self.flatten(data_stream)
        data_stream = self.dense(data_stream)
        data_stream = self.reshape(data_stream)
        return {"vpdepth": data_stream}

    def setup(self, run_eagerly: bool = False):
        optimizer = Adam(learning_rate=self.params.learning_rate,
                         beta_1=self.params.beta_1,
                         beta_2=self.params.beta_2,
                         epsilon=self.params.epsilon,
                         name="Adam")
        loss = {"vpdepth": mean_squared_error}
        self.compile(optimizer=optimizer,
                     loss=loss,
                     run_eagerly=run_eagerly)


def build_conv_block(qties_filters: list):
    layers = []
    for qty_filters in qties_filters:
        conv_2d = Conv2D(filters=qty_filters, kernel_size=(3, 3),
                         padding='same')
        layers.append(conv_2d)
        activation = LeakyReLU()
        layers.append(activation)
    pooling = MaxPool2D(pool_size=(2, 2))
    layers.append(pooling)
    return layers
