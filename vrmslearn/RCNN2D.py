#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 2D prediction vp in depth
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv3D, Conv2D, LeakyReLU, LSTM, Permute, Input,
)
from tensorflow.keras.backend import (
    max as reduce_max, sum as reduce_sum, squeeze, reshape,
)

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

        self.inputs = self.build_inputs()
        self.outputs = self.build_network()
        self.model = Model(
            inputs=self.inputs,
            outputs=self.outputs,
            name="RCNN2D",
        )

        # `RCNN2D` has the same interface as a keras `Model`, but subclassing
        # is avoided by using the functional API. This is necessary for model
        # intelligibility.
        self.compile = self.model.compile
        self.fit = self.model.fit
        self.predict = self.model.predict

    def build_inputs(self):
        with tf.name_scope('Inputs'):
            inputs = Input(
                shape=self.input_size,
                batch_size=self.batch_size,
                dtype=tf.float32,
            )

        return inputs

    def build_network(self):
        outputs = {}
        data_stream = self.scale_inputs(self.inputs)

        with tf.name_scope('Encoder'):
            KERNELS = [
                [15, 1, 1],
                [1, 9, 1],
                [15, 1, 1],
                [1, 9, 1],
            ]
            QTIES_FILTERS = [16, 16, 32, 32]
            for i, (kernel, qty_filters) in (
                        enumerate(zip(KERNELS, QTIES_FILTERS))
                    ):
                with tf.name_scope('CNN_' + str(i)):
                    conv3d = Conv3D(
                        qty_filters,
                        kernel,
                        padding='same',
                    )
                    data_stream = conv3d(data_stream)
                    data_stream = LeakyReLU()(data_stream)
        self.output_encoder = data_stream

        with tf.name_scope('Time_RCNN'):
            for _ in range(7):
                conv3d = Conv3D(
                    32,
                    [15, 3, 1],
                    padding='same',
                )
                data_stream = conv3d(data_stream)
                data_stream = LeakyReLU()(data_stream)
        self.output_time_rcnn = data_stream

        # TODO test mean global pooling, compared to RCNN used in the 1D article
        with tf.name_scope('Global_pooling'):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)
        self.output_global_pooling = data_stream

        if 'ref' in self.out_names:
            with tf.name_scope('Decode_refevent'):
                conv2d = Conv2D(
                    2,
                    [1, 1],
                    padding='same',
                )
                outputs['ref'] = conv2d(data_stream)

        data_stream = Permute((2, 1, 3))(data_stream)
        batches, shots, timesteps, filter_dim = data_stream.get_shape()
        data_stream = reshape(
            data_stream,
            [batches*shots, timesteps, filter_dim],
        )

        with tf.name_scope('RNN_vrms'):
            UNITS = 200
            lstm = LSTM(UNITS, return_sequences=True)
            data_stream = lstm(data_stream)
            self.rnn_vrms_out = data_stream

        if 'vrms' in self.out_names:
            with tf.name_scope('Decode_rms'):
                decode_rms = reshape(
                    data_stream,
                    [batches, shots, timesteps, UNITS],
                )
                decode_rms = Permute((2, 1, 3))(decode_rms)
                conv2d = Conv2D(
                    1,
                    [1, 1],
                    padding='same',
                )
                decode_rms = conv2d(decode_rms)
                outputs['vrms'] = decode_rms

        with tf.name_scope('RNN_vint'):
            lstm = LSTM(UNITS, return_sequences=True)
            data_stream = lstm(data_stream)
            self.rnn_vint_out = data_stream

        if 'vint' in self.out_names:
            with tf.name_scope('Decode_vint'):
                decode_int = reshape(
                    data_stream,
                    [batches, shots, timesteps, UNITS],
                )
                decode_int = Permute((2, 1, 3))(decode_int)
                conv2d = Conv2D(
                    1,
                    [1, 1],
                    padding='same',
                )
                decode_int = conv2d(decode_int)
                outputs['vint'] = decode_int

        #TODO test depth predicitons
        #TODO assess if 1D predictions in depth should be performed before 2D

        if 'vdepth' in self.out_names:
            with tf.name_scope('RNN_vdepth'):
                lstm = LSTM(UNITS, return_sequences=True)
                data_stream = lstm(data_stream)
                self.rnn_vdepth_out = data_stream

            data_stream = reshape(
                data_stream,
                [batches, shots, timesteps, UNITS],
            )
            data_stream = Permute((2, 1, 3))(data_stream)

            KERNELS = [[1, 3], [1, 3], [1, 3], [1, 3]]
            QTIES_FILTERS = [
                2 * UNITS,
                2 * UNITS,
                UNITS,
                UNITS//2,
            ]
            with tf.name_scope('Decode_vp'):
                for i, (kernel, qty_filters) in (
                            enumerate(zip(KERNELS, QTIES_FILTERS))
                        ):
                    with tf.name_scope('CNN_' + str(i)):
                        conv_2d = Conv2D(
                            qty_filters,
                            kernel,
                            padding='same',
                        )
                        data_stream = conv_2d(data_stream)
                        data_stream = LeakyReLU()(data_stream)
                conv_2d = Conv2D(
                    1,
                    [1, 3],
                    padding='same',
                )
                data_stream = conv_2d(data_stream)
                data_stream = data_stream[:, :self.depth_size]
                outputs['vdepth'] = data_stream

        return [outputs[out] for out in self.out_names]

    def scale_inputs(self, inputs):
        """
        Scale each trace to its RMS value, and each CMP to its RMS.

        @params:

        @returns:
        scaled (tf.tensor)  : The scaled input data
        """
        scaled = inputs / (
            tf.sqrt(reduce_sum(inputs ** 2, axis=[1],  keepdims=True))
            + np.finfo(np.float32).eps
        )
        scaled = 1000 * scaled / tf.reduce_max(
            scaled, axis=[1, 2, 3], keepdims=True,
        )
        return scaled
