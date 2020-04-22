#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 2D prediction vp in depth
"""
import numpy as np
import tensorflow as tf
from .conv_lstm.cell import ConvLSTMCell

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

        weights = [tf.Variable(tf.random.normal([15, 1, 1, 1, 16], stddev=1e-1),
                               name='w1'),
                   tf.Variable(tf.random.normal([1, 9, 1, 16, 16], stddev=1e-1),
                               name='w2'),
                   tf.Variable(tf.random.normal([15, 1, 1, 16, 32], stddev=1e-1),
                               name='w3'),
                   tf.Variable(tf.random.normal([1, 9, 1, 32, 32], stddev=1e-1),
                               name='w4'),
                   tf.Variable(tf.random.normal([15, 3, 1, 32, 32], stddev=1e-2),
                               name='w5')]

        biases = [tf.Variable(tf.zeros([16]), name='b1'),
                  tf.Variable(tf.zeros([16]), name='b2'),
                  tf.Variable(tf.zeros([32]), name='b3'),
                  tf.Variable(tf.zeros([32]), name='b4'),
                  tf.Variable(tf.zeros([32]), name='b5')]

        data_stream = self.input_scaled
        allout = [self.input_scaled]
        with tf.name_scope('Encoder'):
            for ii in range(len(weights) - 1):
                with tf.name_scope('CNN_' + str(ii)):
                    data_stream = tf.nn.relu(
                        tf.nn.conv3d(data_stream,
                                     weights[ii],
                                     strides=[1, 1, 1, 1, 1],
                                     padding='SAME') + biases[ii])
                    allout.append(data_stream)
        self.output_encoder = data_stream

        with tf.name_scope('Time_RCNN'):
            for ii in range(7):
                data_stream = tf.nn.relu(
                    tf.nn.conv3d(data_stream,
                                 weights[-1],
                                 strides=[1, 1, 1, 1, 1],
                                 padding='SAME') + biases[-2])
                allout.append(data_stream)

        self.output_time_rcnn = data_stream

        with tf.name_scope('Global_pooling'):
            # TODO test mean global pooling
            data_stream = reduce_max(data_stream, axis=[2], keepdims=False)
        self.output_global_pooling = data_stream

        if 'ref' in self.out_names:
            output_size = int(data_stream.get_shape()[-1])
            with tf.name_scope('Decode_refevent'):
                decode_refw = tf.Variable(
                    initial_value=tf.random.normal([1, 1, output_size, 2],
                                                   stddev=1e-4),
                    name='decode_ref')
                decode_ref = tf.nn.conv2d(data_stream, decode_refw,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME')
                outputs['ref'] = decode_ref

        shape_space = [int(el) for el in data_stream.get_shape()]
        data_stream = tf.reshape(data_stream,
                                 [-1, shape_space[1], shape_space[-1]])
        if 'vrms' in self.out_names:
            with tf.name_scope('RNN_vrms'):
                rnn_hidden = 200
                cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden,
                                               state_is_tuple=True,
                                               use_peepholes=self.use_peepholes)
                state0 = cell.zero_state(data_stream.get_shape()[0], tf.float32)
                data_stream, rnn_states = tf.nn.dynamic_rnn(cell, data_stream,
                                                            initial_state=state0,
                                                            time_major=False,
                                                            scope="rnn_vrms")
                self.rnn_vrms_out = data_stream

            with tf.name_scope('Decode_rms'):
                output_size = int(data_stream.get_shape()[-1])
                decode_rmsw = tf.Variable(
                    initial_value=tf.random.normal([1, 1, output_size, 1], stddev=1e-4),
                                                   name='decode_rms')
                shape_space[-1] = output_size
                decode_rms = tf.reshape(data_stream, shape_space)
                decode_rms = tf.nn.conv2d(decode_rms, decode_rmsw,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME')
                outputs['vrms'] = tf.squeeze(decode_rms, axis=-1)

        if 'vint' in self.out_names:
            with tf.name_scope('RNN_vint'):
                    cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden, state_is_tuple=True,
                                                   use_peepholes=self.use_peepholes)
                    state0 = cell.zero_state(data_stream.get_shape()[0], tf.float32)
                    data_stream, rnn_states = tf.nn.dynamic_rnn(cell, data_stream,
                                                                initial_state=state0,
                                                                time_major=False,
                                                                scope="rnn_vint")
                    self.rnn_vint_out = data_stream

            with tf.name_scope('Decode_vint'):
                output_size = int(data_stream.get_shape()[-1])
                decode_vintw = tf.Variable(
                    initial_value=tf.random.normal([1, 1, output_size, 1], stddev=1e-4),
                                                   name='decode_vint')
                shape_space[-1] = output_size
                decode_vint = tf.reshape(data_stream, shape_space)
                decode_vint = tf.nn.conv2d(decode_vint, decode_vintw,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME')
                outputs['vint'] = tf.squeeze(decode_vint, axis=-1)

        if 'vdepth' in self.out_names:
            with tf.name_scope('RNN_vdepth'):
                    cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden,
                                                   state_is_tuple=True,
                                                   use_peepholes=self.use_peepholes)
                    state0 = cell.zero_state(data_stream.get_shape()[0],
                                             tf.float32)
                    data_stream, rnn_states = tf.nn.dynamic_rnn(cell, data_stream,
                                                                initial_state=state0,
                                                                time_major=False,
                                                                scope="rnn_vdepth")
                    self.rnn_vdepth_out = data_stream

            c = int(data_stream.get_shape()[-1])
            shape_space[-1] = c
            data_stream = tf.reshape(data_stream, shape_space)
            weights = [tf.Variable(tf.random.normal([1, 3, c, 2 * c],
                                                    stddev=1e-1), name='wd1'),
                       tf.Variable(tf.random.normal([1, 3, 2 * c, 2 * c],
                                                    stddev=1e-1), name='wd2'),
                       tf.Variable(tf.random.normal([1, 3, 2 * c, c],
                                                    stddev=1e-1), name='wd3'),
                       tf.Variable(tf.random.normal([1, 3, c, int(c/2)],
                                                    stddev=1e-1), name='wd4'),
                       tf.Variable(tf.random.normal([1, 3, int(c/2), 1],
                                                    stddev=1e-1), name='wd5')]

            biases = [tf.Variable(tf.zeros([2*c]), name='bd1'),
                      tf.Variable(tf.zeros([2*c]), name='bd2'),
                      tf.Variable(tf.zeros([c]), name='bd3'),
                      tf.Variable(tf.zeros([int(c/2)]), name='bd4')]

            with tf.name_scope('Decode_vp'):
                for ii in range(len(weights) - 1):
                    with tf.name_scope('CNN_' + str(ii)):
                        data_stream = tf.nn.relu(
                            tf.nn.conv2d(data_stream,
                                         weights[ii],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME') + biases[ii])
                        allout.append(data_stream)
                data_stream = tf.nn.conv2d(data_stream, weights[-1],
                                           strides=[1, 1, 1, 1],
                                           padding='SAME')
                data_stream = tf.squeeze(data_stream, axis=3)
                data_stream = data_stream[:, :self.depth_size, :]
            outputs['vdepth'] = tf.squeeze(data_stream, axis=-1)

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
                    if lbl is 'vdepth':
                        weight = self.weights['dweight']
                    else:
                        weight = self.weights['tweight']
                    # Calculate mean squared error of vp velocity
                    output = self.outputs[lbl]
                    label = self.labels[lbl]
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

# Tensorflow compatibility
def reduce_sum(a, axis=None, keepdims=True):
    if tf.__version__ == '1.2.0':
        return tf.reduce_sum(a, axis=axis, keep_dims=keepdims)
    else:
        return tf.reduce_sum(a, axis=axis, keepdims=keepdims)

def reduce_max(a, axis=None, keepdims=True):
    if tf.__version__ == '1.2.0':
        return tf.reduce_max(a, axis=axis, keep_dims=keepdims)
    else:
        return tf.reduce_max(a, axis=axis, keepdims=keepdims)
