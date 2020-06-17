#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 2D prediction vp in depth
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, Conv2D, LeakyReLU, LSTM, Permute,
)
from tensorflow.keras.backend import (
    max as reduce_max, sum as reduce_sum, squeeze, reshape,
)

class RCNN2D(object):
    """
    This class build a NN based on recursive CNN and LSTM that can predict
    2D vp velocity
    """

    def __init__(self,
                 input_size: list = (0, 0, 0),
                 depth_size: int = 0,
                 out_names: list = ('ref', 'vrms', 'vint', 'vdepth'),
                 loss_scales: dict = None,
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

        if loss_scales is None:
            loss_scales = {'ref': 1.0}

        outs = ('ref', 'vrms', 'vint', 'vdepth')
        for l in out_names:
            if l not in outs:
                raise ValueError("out_names should be from " + str(outs))
        self.input_size = input_size
        self.depth_size = depth_size
        self.out_names = out_names
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.use_peepholes = use_peepholes
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self.input, self.labels, self.weights = self.generate_io()
            self.feed_dict = {'input': self.input,
                              **self.labels,
                              **self.weights}
            self.input_scaled = self.scale_input()
            self.outputs = self.build_neural_net()
            self.loss, self.losses = self.define_loss(alpha=alpha, beta=beta,
                                                      scales=loss_scales)


    def generate_io(self):
        """
        This method creates the input nodes.

        @params:

        @returns:
        input_data (tf.tensor)  : Placeholder of CMP gather.
        label_vp (tf.placeholder) : Placeholder of RMS velocity labels.
        weights (tf.placeholder) : Placeholder of time weights
        label_ref (tf.placeholder) : Placeholder of primary reflection labels.
        label_vint (tf.placeholder) : Placeholder of interval velocity labels.
        """

        with tf.name_scope('Inputs'):
            # Create placeholder for input
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[self.batch_size,
                                               self.input_size[0],
                                               self.input_size[1],
                                               self.input_size[2],
                                               1],
                                        name='data')
            labels = {}
            weights = {}
            timeout = False
            for label in ['ref', 'vrms', 'vint']:
                if label in self.out_names:
                    labels[label] = tf.placeholder(dtype=tf.float32,
                                                   shape=[self.batch_size,
                                                          self.input_size[0],
                                                          self.input_size[2]],
                                                   name=label)
                    timeout=True
            if timeout:
                weights['tweight'] = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size,
                                                         self.input_size[0],
                                                         self.input_size[2]],
                                                  name='tweight')

            if 'vdepth' in self.out_names:
                labels['vdepth'] = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size,
                                                         self.depth_size,
                                                         self.input_size[2]],
                                                  name='vdepth')
                weights['dweight'] = tf.placeholder(dtype=tf.float32,
                                                   shape=[self.batch_size,
                                                          self.depth_size,
                                                          self.input_size[2]],
                                                   name='dweight')

        return input_data, labels, weights

    def scale_input(self):
        """
        Scale each trace to its RMS value, and each CMP to its RMS.

        @params:

        @returns:
        scaled (tf.tensor)  : The scaled input data
        """
        scaled = self.input / (tf.sqrt(reduce_sum(self.input ** 2, axis=[1],
                                       keepdims=True))
                               + np.finfo(np.float32).eps)

        scaled = 1000*scaled / tf.reduce_max(scaled, axis=[1, 2, 3],
                                             keepdims=True)

        return scaled

    def build_neural_net(self):
        """
        This method build the neural net in Tensorflow

        @params:

        @returns:
        output_vp (tf.tensor) : The vp velocity predictions
        """

        outputs = {}
        data_stream = self.input_scaled

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
                outputs['vrms'] = squeeze(decode_rms, axis=-1)

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
                outputs['vint'] = squeeze(decode_int, axis=-1)

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
                data_stream = squeeze(data_stream, axis=3)
                data_stream = data_stream[:, :self.depth_size, :]
                outputs['vdepth'] = data_stream

        return outputs

    def define_loss(self, alpha=0.2, beta=0.1, scales={'ref': 1.0}):
        """
        This method creates a node to compute the loss function.
        The loss is normalized.

        @params:

        @returns:
        loss (tf.tensor) : Output of node calculating loss.
        """
        with tf.name_scope("Loss_Function"):

            losses = {}
            loss = 0
            fact1 = (1 - alpha - beta)
            for lbl in scales:
                if lbl == 'ref':
                    #  Logistic regression of zero offset time of reflections
                    weightsr = tf.expand_dims(self.weights['tweight'], -1)
                    # if self.with_masking:
                    #     weightsr = tf.expand_dims(self.weights['wtime'], -1)
                    # else:
                    #     weightsr = 1.0
                    preds = self.outputs[lbl] * weightsr
                    labels = tf.one_hot(tf.cast(self.labels[lbl], tf.int32), 2) * weightsr

                    losses[lbl] = [scales[lbl] * tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds,
                                                                    labels=labels))]
                    loss += losses[lbl][-1]
                elif lbl in self.outputs:
                    losses[lbl] = []
                    if lbl == 'vdepth':
                        weight = self.weights['dweight']
                        weight = weight[:, :self.input_size[0], :]
                    else:
                        weight = self.weights['tweight']
                    # Calculate mean squared error of vp velocity
                    output = self.outputs[lbl]
                    label = self.labels[lbl]
                    if lbl == 'vdepth':
                        label = label[:, :self.input_size[0], :]
                    if fact1 > 0:
                        num = tf.reduce_sum(weight * (label - output) ** 2)
                        den = tf.reduce_sum(weight * label ** 2)
                        losses[lbl].append(scales[lbl] * fact1 * num / den)
                        loss += losses[lbl][-1]
                    #  Calculate mean squared error of the z derivative
                    if alpha > 0:
                        dlabel = label[:, 1:, :] - label[:, :-1, :]
                        dout = output[:, 1:, :] - output[:, :-1, :]
                        num = tf.reduce_sum(weight[:, :-1, :]*(dlabel - dout) ** 2)
                        den = tf.reduce_sum(weight[:, :-1, :]*dlabel ** 2 + 0.000001)
                        losses[lbl].append(scales[lbl] * alpha * num / den)
                        loss += losses[lbl][-1]
                    # Minimize interval velocity gradient (blocky inversion)
                    if beta > 0:
                        num = tf.norm((output[:, 1:, :]
                                       - output[:, :-1, :]), ord=1)
                        num += tf.norm((output[:, :, 1:]
                                       - output[:, :, :-1]), ord=1)
                        den = tf.norm(output, ord=1) / 0.02
                        losses[lbl].append(scales[lbl] * beta * num / den)
                        loss += losses[lbl][-1]

            tf.summary.scalar("loss", loss)
        return loss, losses
