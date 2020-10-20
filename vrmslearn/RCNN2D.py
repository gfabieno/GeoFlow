#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to build the neural network for 2D prediction vp in depth
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Conv3D, Conv2D, LeakyReLU, LSTM, Permute,
                                     Input)
from tensorflow.keras.backend import (max as reduce_max, sum as reduce_sum,
                                      reshape, cumsum, arange)

from vrmslearn.Sequence import OUTS
from vrmslearn.Case import Case


class RCNN2D:
    """
    This class build a NN based on recursive CNN and LSTM that can predict
    2D vp velocity
    """

    def __init__(self,
                 input_size: list = (0, 0, 0),
                 out_names: list = ('ref', 'vrms', 'vint', 'vdepth'),
                 batch_size: int = 1,
                 alpha: float = 0,
                 beta: float = 0,
                 restore_from: str = None,
                 freeze_to: str = None,
                 case: Case = None):
        """
        Build the neural net in tensorflow, along the cost function

        @params:
        input_size (list): the size of the CMP [NT, NX, NCMP]
        labels (list): List of the label names to predict from
                       ['ref', 'vrms', 'vint', 'vdepth']
        batch_size (int): Number of examples in a batch
        alpha (float): Fraction of the loss dedicated match the time derivative
        beta (float): Fraction of the loss dedicated minimize model derivative
        ndim (int): Number of dimensions (2 for layered models, 3 for dipping)
        restore_from (str): Checkpoint file from which to initialize parameters
        freeze_to (str): A label name. All layers before this label's decoder
                         will not be trainable.
        case (Case): Constants `vmin`, `vmax`, `dh`, `dt`, `resampling`,
                     `tdelay`, `nz`, `source_depth` and `receiver_depth` are
                     used for time-to-depth conversion. Required if `'vdepth'`
                     is in `out_names`.
        """
        for l in out_names:
            if l not in OUTS:
                raise ValueError(f"`out_names` should be from {OUTS}")
        self.out_names = [out for out in OUTS if out in out_names]

        self.input_size = input_size
        self.batch_size = batch_size

        if 'vdepth' in out_names and case is None:
            raise ValueError("Time-to-depth conversion requires `case`.")
        self.case = case

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.inputs = self.build_inputs()
            self.outputs = self.build_network(freeze_to)
            self.model = Model(inputs=self.inputs,
                               outputs=self.outputs,
                               name="RCNN2D")

            # `RCNN2D` has the same interface as a keras `Model`, but
            # subclassing is avoided by using the functional API. This is
            # necessary for model intelligibility.
            self.compile = self.model.compile
            self.fit = self.model.fit
            self.predict = self.model.predict
            self.set_weights = self.model.set_weights
            self.get_weights = self.model.get_weights
            self.layers = self.model.layers

            if restore_from is not None:
                self.load_weights(restore_from)

    def build_inputs(self):
        inputs = Input(shape=self.input_size,
                       batch_size=self.batch_size,
                       dtype=tf.float32)
        return inputs

    def build_network(self, freeze_to):
        is_1d = self.input_size[2] == 1

        outputs = {}

        data_stream = self.scale_inputs(self.inputs)

        encoder = build_encoder(kernels=[[15, 1, 1],
                                         [1, 9, 1],
                                         [15, 1, 1],
                                         [1, 9, 1]],
                                qties_filters=[16, 16, 32, 32])
        if freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            encoder.trainable = False
        data_stream = encoder(data_stream)

        time_rcnn = build_rcnn(reps=7,
                               kernel=[15, 3, 1],
                               qty_filters=32,
                               input_shape=data_stream.shape,
                               batch_size=self.batch_size,
                               name="time_rcnn")
        if freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            time_rcnn.trainable = False
        data_stream = time_rcnn(data_stream)

        with tf.name_scope("global_pooling"):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)

        if 'ref' in self.out_names:
            conv_2d = Conv2D(2, [1, 1], padding='same', name="ref")
            outputs['ref'] = conv_2d(data_stream)

        rnn_vrms = build_rnn(units=200, input_shape=data_stream.shape,
                             batch_size=self.batch_size, name="rnn_vrms")
        if freeze_to in ['vrms', 'vint', 'vdepth']:
            rnn_vrms.trainable = False
        data_stream = rnn_vrms(data_stream)

        if 'vrms' in self.out_names:
            if is_1d:
                kernel = [1, 1]
            else:
                kernel = [1, 5]
            conv_2d = Conv2D(1, kernel, padding='same', name="vrms")
            outputs['vrms'] = conv_2d(data_stream)

        rnn_vint = build_rnn(units=200, input_shape=data_stream.shape,
                             batch_size=self.batch_size, name="rnn_vint")
        if freeze_to in ['vint', 'vdepth']:
            rnn_vint.trainable = False
        data_stream = rnn_vint(data_stream)

        if 'vint' in self.out_names:
            if is_1d:
                kernel = [1, 1]
            else:
                kernel = [1, 5]
            conv_2d = Conv2D(1, kernel, padding='same', name="vint")
            outputs['vint'] = conv_2d(data_stream)

        if 'vdepth' in self.out_names:
            vint = outputs['vint']
            time_to_depth = build_time_to_depth_converter(self.case,
                                                          vint.shape[1:],
                                                          self.batch_size,
                                                          name="vdepth")
            vdepth = time_to_depth(vint)
            outputs['vdepth'] = vdepth

        return [outputs[out] for out in self.out_names]

    def scale_inputs(self, inputs):
        """
        Scale each trace to its RMS value, and each shot to its RMS.

        @params:

        @returns:
        scaled (tf.tensor)  : The scaled input data
        """
        trace_rms = tf.sqrt(reduce_sum(inputs**2, axis=1, keepdims=True))
        scaled = inputs / (trace_rms+np.finfo(np.float32).eps)
        shot_max = tf.reduce_max(scaled, axis=[1, 2], keepdims=True)
        scaled = scaled / shot_max
        return scaled

    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        """
        Load weights into the model and broadcast 1D to 2D correctly.

        :param filepath: String, path to the weights file to load. For weight
                         files in TensorFlow format, this is the file prefix
                         (the same as was passed to save_weights).
        :param by_name: Boolean, whether to load weights by name or by
                        topological order. Only topological loading is
                        supported for weight files in TensorFlow format. This
                        is not implemented in this model subclassing.
        :param skip_mismatch: Boolean, whether to skip loading of layers where
                              there is a mismatch in the number of weights, or
                              a mismatch in the shape of the weight (only valid
                              when by_name=True). This is not implemented in
                              this model subclassing.
        """
        if by_name or skip_mismatch:
            raise NotImplementedError

        model = tf.keras.models.load_model(filepath, compile=False)
        weights = self.get_weights()
        new_weights = model.get_weights()
        layer_names = [l.name for l in model.layers]
        # Backward compatibility with models prior to commit 6cdd52d.
        if "rnn_vdepth" in layer_names:
            del new_weights[16:19]
            del new_weights[-2:]
            new_weights.insert(-6, new_weights[-2])
            new_weights.insert(-6, new_weights[-1])
            del new_weights[-2:]
        for i, (layer, new_layer) in enumerate(zip(weights, new_weights)):
            if layer.shape != new_layer.shape:
                assert_broadcastable(layer, new_layer,
                                     "Weights dimensions are not compatible.")
                mismatches = np.not_equal(layer.shape, new_layer.shape)
                mismatch_idx = np.nonzero(mismatches)[0]
                if len(mismatch_idx) != 1:
                    raise NotImplementedError("No loading strategy is "
                                              "implemented for weights with "
                                              "more than 1 mismatching "
                                              "dimension.")

                # Extend `new_layer` in the missing dimension and rescale it
                # to account for the addition of duplicated layers.
                mismatch_idx = mismatch_idx[0]
                repeats = layer.shape[mismatch_idx]
                new_layer = np.repeat(new_layer, repeats, axis=mismatch_idx)
                new_layer /= repeats
                new_layer += np.random.uniform(0, np.amax(new_layer)*1E-2,
                                               size=new_layer.shape)
                new_weights[i] = new_layer
        self.set_weights(new_weights)


def build_encoder(kernels, qties_filters, dilation_rates, name="encoder"):
    encoder = Sequential(name=name)
    for kernel, qty_filters, dilation_rate in zip(kernels, qties_filters,
                                                  dilation_rates):
        encoder.add(Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                           padding='same'))
        encoder.add(LeakyReLU())
    return encoder


def build_rcnn(reps, kernel, qty_filters, dilation_rate, input_shape,
               batch_size, input_dtype=tf.float32, name="rcnn"):
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size,
                  dtype=input_dtype)

    data_stream = input
    conv_3d = Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                     padding='same')
    activation = LeakyReLU()
    for _ in range(reps):
        data_stream = conv_3d(data_stream)
        data_stream = activation(data_stream)
    rcnn = Model(inputs=input, outputs=data_stream, name=name)
    return rcnn


def build_rnn(units, input_shape, batch_size, input_dtype=tf.float32,
              name="rnn"):
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size,
                  dtype=input_dtype)
    data_stream = Permute((2, 1, 3))(input)
    batches, shots, timesteps, filter_dim = data_stream.get_shape()
    data_stream = reshape(data_stream,
                          [batches*shots, timesteps, filter_dim])
    lstm = LSTM(units, return_sequences=True)
    data_stream = lstm(data_stream)
    data_stream = reshape(data_stream,
                          [batches, shots, timesteps, units])
    data_stream = Permute((2, 1, 3))(data_stream)

    rnn = Model(inputs=input, outputs=data_stream, name=name)
    return rnn


def assert_broadcastable(arr1, arr2, message=None):
    try:
        np.broadcast(arr1, arr2)
    except ValueError:
        if message is None:
            message = "Arrays are not compatible for broadcasting."
        raise AssertionError(message)


def build_time_to_depth_converter(case, input_shape, batch_size,
                                  input_dtype=tf.float32,
                                  name="time_to_depth_converter"):
    vmax = case.model.vp_max
    vmin = case.model.vp_min
    dh = case.model.dh
    dt = case.acquire.dt
    resampling = case.acquire.resampling
    tdelay = case.acquire.tdelay
    tdelay = round(tdelay / (dt*resampling))  # Convert to unitless time steps.
    nz = case.model.NZ
    source_depth = case.acquire.source_depth
    max_depth = nz - source_depth / dh

    vint = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    actual_vint = vint*(vmax-vmin) + vmin
    # Convert to unitless quantity of grid cells.
    depth_intervals = actual_vint * dt * resampling / (dh*2)
    depths = cumsum(depth_intervals, axis=1)
    depth_delay = reduce_sum(depth_intervals[:, :tdelay], axis=1,
                             keepdims=True)
    depths -= depth_delay
    target_depths = arange(max_depth, dtype=tf.float32)
    vdepth = interp_nearest(x=target_depths, x_ref=depths, y_ref=vint, axis=1)

    time_to_depth_converter = Model(inputs=vint, outputs=vdepth, name=name)
    return time_to_depth_converter


def interp_nearest(x, x_ref, y_ref, axis=0):
    """TensorFlow implementation of 1D nearest neighbors interpolation.

    :param x: Positions of the new sampled data points. Has one dimension.
    :param x_ref: Reference data points. `x_ref` has a first dimension of
        arbitrary length. Other dimensions are treated independently.
    :param y_ref: Reference data points. `y_ref` has the same shape as `x_ref`.
    :param axis: Dimension along which interpolation is executed.
    """
    new_dims = iter([axis, 0])
    # Create a list where `axis` and `0` are interchanged.
    permutation = [dim if dim not in [axis, 0] else next(new_dims)
                   for dim in range(tf.rank(x_ref))]
    x_ref = tf.transpose(x_ref, permutation)
    y_ref = tf.transpose(y_ref, permutation)

    x_ref = tf.expand_dims(x_ref, axis=0)
    while tf.rank(x) != tf.rank(x_ref):
        x = tf.expand_dims(x, axis=-1)
    distances = tf.abs(x_ref-x)
    nearest_neighbor = tf.argmin(distances, axis=1, output_type=tf.int32)

    grid = tf.meshgrid(*[tf.range(dim) for dim in nearest_neighbor.shape],
                       indexing="ij")
    nearest_neighbor = tf.reshape(nearest_neighbor, [-1])
    grid = [tf.reshape(t, [-1]) for t in grid[1:]]
    idx = [nearest_neighbor, *grid]
    idx = tf.transpose(idx, (1, 0))
    y = tf.gather_nd(y_ref, idx)
    y = tf.reshape(y, [-1, *y_ref.shape[1:]])

    y = tf.transpose(y, permutation)
    return y
