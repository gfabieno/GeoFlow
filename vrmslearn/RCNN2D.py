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
                 input_size: list = [0, 0, 0],
                 label_size: list = [0, 0],
                 batch_size: int = 1,
                 alpha: float = 0,
                 beta: float = 0,
                 use_peepholes: bool = False):
        """
        Build the neural net in tensorflow, along the cost function

        @params:
        input_size (list): the size of the CMP [NT, NX, NCMP]
        label_size (list): the size of the labels [NZ, NX]
        batch_size (int): Number of examples in a batch
        alpha (float): Fraction of the loss dedicated match the time derivative
        beta (float): Fraction of the loss dedicated minimize model derivative
        use_peepholes (bool): If true, use peephole LSTM

        @returns:
        """

        self.input_size = input_size
        self.label_size = label_size
        if input_size[-1] != label_size[-1]:
            raise ValueError("Last dimension of labels and inputs should match")
        self.graph = tf.Graph()
        self.feed_dict = []
        self.batch_size = batch_size
        self.use_peepholes = use_peepholes
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self.input, self.label_vp, self.weights = self.generate_io()
            self.feed_dict = [self.input,
                              self.label_vp,
                              self.weights]
            self.input_scaled = self.scale_input()
            self.output_vp = self.build_neural_net()
            self.loss = self.define_loss(alpha=alpha,
                                         beta=beta)


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

            label_vp = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batch_size,
                                             self.label_size[0],
                                             self.label_size[1]],
                                      name='vp')

            weights = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size,
                                            self.label_size[0],
                                            self.label_size[1]],
                                     name='weigths')


        return input_data, label_vp, weights

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

        weights = [tf.Variable(tf.random_normal([15, 1, 1, 1, 16], stddev=1e-1),
                               name='w1'),
                   tf.Variable(tf.random_normal([1, 9, 1, 16, 16], stddev=1e-1),
                               name='w2'),
                   tf.Variable(tf.random_normal([1, 1, 9, 16, 16], stddev=1e-1),
                               name='w3'),
                   tf.Variable(tf.random_normal([15, 1, 1, 16, 32], stddev=1e-1),
                               name='w4'),
                   tf.Variable(tf.random_normal([1, 9, 1, 32, 32], stddev=1e-1),
                               name='w5'),
                   tf.Variable(tf.random_normal([1, 1, 9, 32, 32], stddev=1e-1),
                               name='w6'),
                   tf.Variable(tf.random_normal([15, 3, 3, 32, 32], stddev=1e-2),
                               name='w7')]

        biases = [tf.Variable(tf.zeros([16]), name='b1'),
                  tf.Variable(tf.zeros([16]), name='b2'),
                  tf.Variable(tf.zeros([16]), name='b3'),
                  tf.Variable(tf.zeros([32]), name='b4'),
                  tf.Variable(tf.zeros([32]), name='b5'),
                  tf.Variable(tf.zeros([32]), name='b6'),
                  tf.Variable(tf.zeros([32]), name='b7')]

        data_stream = self.input_scaled
        allout = [self.input_scaled]
        with tf.name_scope('Encoder'):
            for ii in range(len(weights) - 2):
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
                                 weights[-2],
                                 strides=[1, 1, 1, 1, 1],
                                 padding='SAME') + biases[-2])
                allout.append(data_stream)

        self.output_time_rcnn = data_stream

        with tf.name_scope('Global_pooling'):
            # TODO test mean global pooling
            data_stream = reduce_max(data_stream, axis=[2], keepdims=False)
        self.output_global_pooling = data_stream

        with tf.name_scope('RNN_vp'):
            # TODO tests different RNN (CONV_LSTM, GRU, bidrectional)
            shape = [data_stream.shape[2].value]
            kernel = [7]
            filters = 16
            # cellf = ConvLSTMCell(shape, filters, kernel,
            #                     peephole=self.use_peepholes)
            # cellb = ConvLSTMCell(shape, filters, kernel,
            #                     peephole=self.use_peepholes)

            data_stream = tf.transpose(data_stream, perm=(0, 2, 1, 3))
            shapei = [d.value for d in data_stream.shape]
            newshape = [shapei[0] * shapei[1]] + shapei[2:]
            data_stream = tf.reshape(data_stream, shape=newshape)

            cellf = tf.nn.rnn_cell.LSTMCell(filters, state_is_tuple=True,
                                            use_peepholes=self.use_peepholes)
            tf.nn.dynamic_rnn(cellf, data_stream, dtype=data_stream.dtype)
            # cellb = tf.nn.rnn_cell.LSTMCell(filters, state_is_tuple=True,
            #                                 use_peepholes=self.use_peepholes)
            # data_stream, rnn_states = tf.nn.bidirectional_dynamic_rnn(cellf,
            #                                                           cellb,
            #                                                           data_stream,
            #                                                           dtype=data_stream.dtype,
            #                                                           swap_memory=False)
            # data_stream = tf.concat(data_stream, 2)
            data_stream = tf.reshape(data_stream, shape=shapei)
            data_stream = tf.transpose(data_stream, perm=(0, 2, 1, 3))
            self.rnn_out = data_stream

        with tf.name_scope('Decode_vp'):
            output_size = int(data_stream.get_shape()[-1])
            w0 = tf.random_normal([1, 1, output_size, 8], stddev=1e-4)
            wvp1 = tf.Variable(w0,  name='wvp1')
            bvp = tf.Variable(tf.zeros([8]), name='bvp1')
            wvp2 = tf.Variable(tf.random_normal([1, 1, 8, 1],
                                                stddev=1e-4),
                               name='wvp2')
            data_stream = tf.nn.relu(tf.nn.conv2d(data_stream, wvp1,
                                     strides=[1, 1, 1, 1],
                                     padding='SAME') + bvp)
            data_stream = tf.nn.conv2d(data_stream, wvp2,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
            data_stream = tf.squeeze(data_stream, axis=3)
            data_stream = data_stream[:, :self.label_size[0], :]

        return data_stream

    def define_loss(self, alpha=0.2, beta=0.1):
        """
        This method creates a node to compute the loss function.
        The loss is normalized.

        @params:

        @returns:
        loss (tf.tensor) : Output of node calculating loss.
        """
        with tf.name_scope("Loss_Function"):

            losses = []

            fact1 = (1 - alpha - beta)

            # Calculate mean squared error of vp velocity
            if fact1 > 0:
                num = tf.reduce_sum(self.weights*(self.label_vp
                                                  - self.output_vp) ** 2)
                den = tf.reduce_sum(self.weights*self.label_vp ** 2)
                losses.append(fact1 * num / den)

            #  Calculate mean squared error of the derivative of the continuous
            # rms velocity(normalized)
            if alpha > 0:
                dlabels = self.label_vp[:, 1:, :] - self.label_vp[:, :-1, :]
                dout = self.output_vp[:, 1:, :] - self.output_vp[:, :-1, :]
                num = tf.reduce_sum(self.weights[:, :-1, :]*(dlabels - dout) ** 2)
                den = tf.reduce_sum(self.weights[:, :-1, :]*dlabels ** 2 + 0.000001)
                losses.append(alpha * num / den)

            # Minimize interval velocity gradient (blocky inversion)
            if beta > 0:
                num = tf.norm((self.output_vp[:, 1:, :]
                               - self.output_vp[:, :-1, :]), ord=1)
                num += tf.norm((self.output_vp[:, :, 1:]
                               - self.output_vp[:, :, :-1]), ord=1)
                den = tf.norm(self.output_vp, ord=1) / 0.02
                losses.append(beta * num / den)

            loss = np.sum(losses)

            tf.summary.scalar("loss", loss)
        return loss


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
