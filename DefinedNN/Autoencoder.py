# -*- coding: utf-8 -*-
"""
Build a toy neural network for autoencoding shot gathers.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam

from GeoFlow.NN import Hyperparameters, NN
from GeoFlow.Losses import mean_squared_error


class Hyperparameters(Hyperparameters):
    def __init__(self):
        self.restore_from = None
        self.epochs = 5
        self.steps_per_epoch = 100
        self.batch_size = 10

        # The learning rate.
        self.learning_rate = 8E-4
        # Adam optimizer hyperparameters.
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-5

        # Quantity of convolution filters and kernel shapes of the encoder.
        self.encoder_qties_filters = [8] * 4
        self.encoder_kernels = [[5, 5, 1]] * 4
        # Quantity of convolution filters and kernel shapes of the decoder.
        self.decoder_qties_filters = [8] * 3 + [1]
        self.decoder_kernels = [[5, 5, 1]] * 4


class Autoencoder(NN):
    """
    Autoencode shot gathers.
    """
    toinputs = ["shotgather"]
    tooutputs = ["reconstructed"]

    def build_inputs(self, input_shape):
        shotgather = Input(shape=input_shape,
                           batch_size=self.params.batch_size,
                           dtype=tf.float32)
        filename = Input(shape=[1],
                         batch_size=self.params.batch_size,
                         dtype=tf.string)
        inputs = {"shotgather": shotgather, "filename": filename}
        return inputs

    def build_network(self, inputs: dict):
        self.encoder = []
        for qty_filters, kernel in zip(self.params.encoder_qties_filters,
                                       self.params.encoder_kernels):
            conv_3d = Conv3D(qty_filters, kernel, activation='relu',
                             padding='same')
            self.encoder.append(conv_3d)

        self.decoder = []
        for qty_filters, kernel in zip(self.params.decoder_qties_filters[:-1],
                                       self.params.decoder_kernels[:-1]):
            conv_3d = Conv3DTranspose(qty_filters, kernel, activation='relu',
                                      padding='same')
            self.decoder.append(conv_3d)
        conv_3d = Conv3D(self.params.decoder_qties_filters[-1],
                         self.params.decoder_kernels[-1],
                         activation='sigmoid',
                         padding='same')
        self.decoder.append(conv_3d)

    def call(self, inputs: dict):
        data_stream = inputs["shotgather"]
        for layer in self.encoder:
            data_stream = layer(data_stream)
        for layer in self.decoder:
            data_stream = layer(data_stream)
        return {"reconstructed": data_stream}

    def setup(self, run_eagerly: bool = False):
        optimizer = Adam(learning_rate=self.params.learning_rate,
                         beta_1=self.params.beta_1,
                         beta_2=self.params.beta_2,
                         epsilon=self.params.epsilon,
                         name="Adam")
        loss = {"reconstructed":
                lambda label, output: mean_squared_error(label, output,
                                                         axis=[1, 2])}
        self.compile(optimizer=optimizer,
                     loss=loss,
                     run_eagerly=run_eagerly)
