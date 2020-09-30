#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 2D prediction vp in depth
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv3D, Conv2D, LeakyReLU, LSTM, Permute,
                                     Input)
from tensorflow.keras.backend import (max as reduce_max, sum as reduce_sum,
                                      reshape)

from vrmslearn.Sequence import OUTS


class RCNN2D:
    """
    This class build a NN based on recursive CNN and LSTM that can predict
    2D vp velocity
    """

    def __init__(self,
                 input_size: list = (0, 0, 0),
                 depth_size: int = 0,
                 out_names: list = ('ref', 'vrms', 'vint', 'vdepth'),
                 batch_size: int = 1,
                 alpha: float = 0,
                 beta: float = 0,
                 use_peepholes: bool = False):
        """
        Build the neural net in tensorflow, along the cost function

        @params:
        input_size (list): the size of the CMP [NT, NX, NCMP]
        labels (list): List of the label names to predict from
                       ['ref', 'vrms', 'vint', 'vdepth']
        batch_size (int): Number of examples in a batch
        alpha (float): Fraction of the loss dedicated match the time derivative
        beta (float): Fraction of the loss dedicated minimize model derivative
        use_peepholes (bool): If true, use peephole LSTM
        ndim (int): Number of dimensions (2 for layered models, 3 for dipping)

        @returns:
        """
        for l in out_names:
            if l not in OUTS:
                raise ValueError(f"`out_names` should be from {OUTS}")
        self.out_names = [out for out in OUTS if out in out_names]

        self.input_size = input_size
        self.depth_size = depth_size
        self.batch_size = batch_size
        self.use_peepholes = use_peepholes

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.inputs = self.build_inputs()
            self.outputs = self.build_network()
            self.model = Model(inputs=self.inputs,
                               outputs=self.outputs,
                               name="RCNN2D")

        # `RCNN2D` has the same interface as a keras `Model`, but subclassing
        # is avoided by using the functional API. This is necessary for model
        # intelligibility.
        self.compile = self.model.compile
        self.fit = self.model.fit
        self.load_weights = self.model.load_weights
        self.predict = self.model.predict

    def build_inputs(self):
        with tf.name_scope('Inputs'):
            inputs = Input(shape=self.input_size,
                           batch_size=self.batch_size,
                           dtype=tf.float32)

        return inputs

    def build_network(self):
        is_1d = self.input_size[2] == 1

        outputs = {}

        data_stream = self.scale_inputs(self.inputs)

        encoder = build_encoder(kernels=[[15, 1, 1],
                                         [1, 9, 1],
                                         [15, 1, 5],
                                         [1, 9, 5]],
                                qties_filters=[16, 16, 32, 32])
        data_stream = encoder(data_stream)

        time_rcnn = build_rcnn(reps=7,
                               kernel=[15, 3, 5],
                               qty_filters=32,
                               name="time_rcnn")
        data_stream = time_rcnn(data_stream)

        with tf.name_scope("global_pooling"):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)

        if 'ref' in self.out_names:
            with tf.name_scope("decode_ref"):
                conv_2d = Conv2D(2, [1, 1], padding='same', name="ref")
                outputs['ref'] = conv_2d(data_stream)

        rnn_vrms = build_rnn(units=200, name="rnn_vrms")
        data_stream = rnn_vrms(data_stream)

        if 'vrms' in self.out_names:
            with tf.name_scope("decode_vrms"):
                if is_1d:
                    kernel = [1, 1]
                else:
                    kernel = [1, 5]
                conv_2d = Conv2D(1, kernel, padding='same', name="vrms")
                outputs['vrms'] = conv_2d(data_stream)

        rnn_vint = build_rnn(units=200, name="rnn_vint")
        data_stream = rnn_vint(data_stream)

        if 'vint' in self.out_names:
            with tf.name_scope("decode_vint"):
                if is_1d:
                    kernel = [1, 1]
                else:
                    kernel = [1, 5]
                conv_2d = Conv2D(1, kernel, padding='same', name="vint")
                outputs['vint'] = conv_2d(data_stream)

        data_stream = data_stream[:, :self.depth_size]
        rnn_vdepth = build_rnn(units=200, name="rnn_vdepth")
        data_stream = rnn_vdepth(data_stream)

        if 'vdepth' in self.out_names:
            with tf.name_scope("decode_vdepth"):
                if is_1d:
                    kernel = [1, 1]
                else:
                    kernel = [1, 5]
                conv_2d = Conv2D(1, kernel, padding='same', name="vdepth")
                outputs['vdepth'] = conv_2d(data_stream)

        return [outputs[out] for out in self.out_names]

    def scale_inputs(self, inputs):
        """
        Scale each trace to its RMS value, and each CMP to its RMS.

        @params:

        @returns:
        scaled (tf.tensor)  : The scaled input data
        """
        trace_rms = tf.sqrt(reduce_sum(inputs**2, axis=[1], keepdims=True))
        scaled = inputs / (trace_rms+np.finfo(np.float32).eps)
        cmp_rms = tf.reduce_max(scaled, axis=[1, 2, 3], keepdims=True)
        scaled = 1000 * scaled / cmp_rms
        return scaled


def build_encoder(kernels, qties_filters, name="encoder"):
    def encoder(data_stream):
        with tf.name_scope(name):
            for i, (kernel, qty_filters) in enumerate(zip(kernels,
                                                          qties_filters)):
                with tf.name_scope(f'CNN_{i}'):
                    conv3d = Conv3D(qty_filters,
                                    kernel,
                                    padding='same')
                    data_stream = conv3d(data_stream)
                    data_stream = LeakyReLU()(data_stream)
        return data_stream
    return encoder


def build_rcnn(reps, kernel, qty_filters, name="rcnn"):
    def rcnn(data_stream):
        with tf.name_scope(name):
            for _ in range(reps):
                conv3d = Conv3D(qty_filters,
                                kernel,
                                padding='same')
                data_stream = conv3d(data_stream)
                data_stream = LeakyReLU()(data_stream)
        return data_stream
    return rcnn


def build_rnn(units, name="rnn"):
    def rnn(data_stream):
        with tf.name_scope(name):
            data_stream = Permute((2, 1, 3))(data_stream)
            batches, shots, timesteps, filter_dim = data_stream.get_shape()
            data_stream = reshape(data_stream,
                                  [batches*shots, timesteps, filter_dim])
            lstm = LSTM(units, return_sequences=True)
            data_stream = lstm(data_stream)
            data_stream = reshape(data_stream,
                                  [batches, shots, timesteps, units])
            data_stream = Permute((2, 1, 3))(data_stream)
        return data_stream
    return rnn
