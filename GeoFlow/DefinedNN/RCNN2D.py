# -*- coding: utf-8 -*-
"""
Build a neural network for predicting v_p in 2D and in depth.
"""

from os import getcwd
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import (Conv3D, Conv2D, LSTM, Permute, Input,
                                     ReLU, Dropout, Bidirectional)
from tensorflow.keras.backend import reshape

from GeoFlow.NN import Hyperparameters, NN
from GeoFlow.Losses import ref_loss, v_compound_loss
from GeoFlow.SeismicUtilities import build_time_to_depth_converter


class Hyperparameters(Hyperparameters):
    def __init__(self, is_training=True):
        """
        Build the default hyperparameters for `RCNN2D`.
        """
        self.restore_from = None
        self.epochs = 5
        self.steps_per_epoch = 100
        self.batch_size = 50
        self.seed = None

        # The learning rate.
        self.learning_rate = 8E-4
        # Adam optimizer hyperparameters.
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-5
        # Losses associated with each label.
        self.loss_scales = {'ref': .5, 'vrms': .4, 'vint': .1, 'vdepth': .0}

        # Whether to add noise or not to the data.
        self.add_noise = False

        # A list of NN components.
        self.freeze = []

        # Convolution kernels of the encoder head.
        self.encoder_kernels = [[15, 1, 1],
                                [1, 9, 1],
                                [15, 1, 1],
                                [1, 9, 1]]
        # Quantity of filters per encoder kernel. Must have the same length as
        # `self.encoder_kernels`.
        self.encoder_filters = [16, 16, 32, 32]
        # Diltations of the convolutions in the encoder head.
        self.encoder_dilations = [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]
        # Convolution kernel of the RCNN.
        self.rcnn_kernel = [15, 3, 1]
        # Quantity of filters in the RCNN.
        self.rcnn_filters = 32
        # Dilation of the convolutions in the RCNN.
        self.rcnn_dilation = [1, 1, 1]
        # Kernel of the convolution associated with the `ref` output.
        self.decode_ref_kernel = [1, 1]
        # Kernel of the convolutions with outputs, except `ref`.
        self.decode_kernel = [1, 1]

        # Whether to interleave CNNs in between RNNs or not.
        self.use_cnn = False
        # Convolution kernel of the CNNs between RNNs, a list of length 3.
        self.cnn_kernel = None
        # Quantity of filters of the CNNs between RNNs, a positive integer.
        self.cnn_filters = None
        # Dilation of the CNNs between RNNs, a list of length 3.
        self.cnn_dilation = None


class RCNN2D(NN):
    """
    Combine a recursive CNN and LSTMs to predict 2D v_p velocity.
    """
    tooutputs = ["ref", "vrms", "vint", "vdepth"]
    toinputs = ["shotgather"]

    def build_network(self, inputs: dict):
        params = self.params
        batch_size = self.params.batch_size

        self.decoder = {}
        self.rnn = {}
        self.cnn = {}

        self.encoder = build_encoder(kernels=params.encoder_kernels,
                                     dilation_rates=params.encoder_dilations,
                                     qties_filters=params.encoder_filters,
                                     input_shape=inputs['shotgather'].shape,
                                     batch_size=batch_size)

        self.rcnn = build_rcnn(reps=7,
                               kernel=params.rcnn_kernel,
                               qty_filters=params.rcnn_filters,
                               dilation_rate=params.rcnn_dilation,
                               input_shape=self.encoder.output_shape,
                               batch_size=batch_size,
                               name="time_rcnn")

        self.rvcnn = build_rcnn(reps=6,
                                kernel=(1, 2, 1),
                                qty_filters=params.rcnn_filters,
                                dilation_rate=(1, 1, 1),
                                strides=(1, 2, 1),
                                padding='valid',
                                input_shape=self.rcnn.output_shape,
                                batch_size=batch_size,
                                name="rvcnn")

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])

        self.decoder['ref'] = Conv2D(1, params.decode_ref_kernel,
                                     padding='same',
                                     activation='sigmoid',
                                     input_shape=shape_after_pooling,
                                     batch_size=batch_size, name="ref")

        self.rnn['vrms'] = build_rnn(units=200,
                                     input_shape=shape_after_pooling,
                                     batch_size=batch_size,
                                     name="rnn_vrms")

        input_shape = self.rnn['vrms'].output_shape
        if params.use_cnn:
            self.cnn['vrms'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vrms")
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vrms'] = Conv2D(1, params.decode_kernel, padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      use_bias=False,
                                      name="vrms")

        self.rnn['vint'] = build_rnn(units=200,
                                     input_shape=input_shape,
                                     batch_size=batch_size,
                                     name="rnn_vint")

        input_shape = self.rnn['vint'].output_shape
        if params.use_cnn:
            self.cnn['vint'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vint")
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vint'] = Conv2D(1, params.decode_kernel, padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      use_bias=False,
                                      name="vint")

        vint_shape = input_shape[1:-1] + (1,)
        self.time_to_depth = build_time_to_depth_converter(self.dataset,
                                                           vint_shape,
                                                           batch_size,
                                                           name="vdepth")

        for layer in params.freeze:
            layer = eval('self.' + layer)
            layer.trainable = False

    def call(self, inputs: dict):
        params = self.params

        outputs = {}

        data_stream = self.encoder(inputs["shotgather"])
        data_stream = self.rcnn(data_stream)
        data_stream = self.rvcnn(data_stream)
        data_stream = data_stream[:, :, 0]

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn['vrms'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vrms'](data_stream)
        outputs['vrms'] = self.decoder['vrms'](data_stream)

        data_stream = self.rnn['vint'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vint'](data_stream)
        outputs['vint'] = self.decoder['vint'](data_stream)

        outputs['vdepth'] = self.time_to_depth(outputs['vint'])

        return {out: outputs[out] for out in self.tooutputs}

    def load_weights(self, filepath: str, by_name: bool = True,
                     skip_mismatch: bool = False):
        """
        Load weights into the model and broadcast 1D to 2D correctly.

        :param filepath: String, path to the weights file to load. For weight
                         files in TensorFlow format, this is the file prefix
                         (the same as was passed to save_weights).
        :param by_name: The only implemented behavior in this model
                        subclassing is `by_name=True`, because it is the only
                        behavior that doesn't cause unexpected results when
                        the model is modified.
        :param skip_mismatch: This is not implemented in this model
                              subclassing, because we broadcast mismatching
                              layers.
        """
        if skip_mismatch or not by_name:
            raise NotImplementedError

        try:
            loaded_model = tf.keras.models.load_model(filepath, compile=False)
        except RuntimeError:
            print(
                "WARNING: Could not use transfer learning on loaded model. "
                "Loading weights as they are."
            )
            filepath = join(getcwd(), filepath, 'variables', 'variables')
            Model.load_weights(self, filepath)
            return
        current_layer_names = [layer.name for layer in self.layers]
        for loaded_layer in loaded_model.layers:
            name = loaded_layer.name
            if name == "rcnn_pooling":  # Allow backwards compatibility.
                name = "rvcnn"
            if name not in current_layer_names:
                print(f"Loading layer {name} skipped.")
                continue
            current_layer = self.get_layer(name)
            current_weights = current_layer.get_weights()
            loaded_weights = loaded_layer.get_weights()
            if len(current_weights) != len(loaded_weights):
                loaded_weights = insert_layers(loaded_weights, current_weights)
            current_weights = broadcast_weights(loaded_weights,
                                                current_weights)
            current_layer.set_weights(current_weights)

    def setup(self, run_eagerly: bool = False):
        losses, losses_weights = self.build_losses()
        optimizer = optimizers.Adam(learning_rate=self.params.learning_rate,
                                    beta_1=self.params.beta_1,
                                    beta_2=self.params.beta_2,
                                    epsilon=self.params.epsilon,
                                    name="Adam")
        self.compile(optimizer=optimizer,
                     loss=losses,
                     loss_weights=losses_weights,
                     run_eagerly=run_eagerly)

    def build_losses(self):
        losses, losses_weights = {}, {}
        for lbl in self.tooutputs:
            if lbl == 'ref':
                losses[lbl] = ref_loss()
            elif lbl == 'vrms':
                losses[lbl] = v_compound_loss(
                    alpha=.0, beta=.0, normalize=True,
                )
            else:
                losses[lbl] = v_compound_loss(normalize=True)
            losses_weights[lbl] = self.params.loss_scales[lbl]

        return losses, losses_weights


def broadcast_weights(loaded_weights: np.ndarray, current_weights: np.ndarray,
                      rescale_current: float = .1):
    """
    Broadcast `loaded_weights` on `current_weights`.

    Extend the loaded weights along one missing dimension and fill additional
    kernel elements with the newly initialized weights, rescaled by
    `rescale_current`. This is equivalent to initializing weights in the lower
    dimensional setting, plus a noise contribution of the newly initialized
    weights `current_weights` off-middle.

    :param loaded_weights: The weights of the recovered checkpoint.
    :type loaded_weights: np.ndarray
    :param current_weights: The current weights of the model, which will be
                            partly overridden.
    :type current_weights: np.ndarray
    :param rescale_current: A scaling factor applied on the newly initialized
                            weights `current_weights`.
    :type rescale_current: float
    """
    for i, (current, loaded) in enumerate(zip(current_weights,
                                              loaded_weights)):
        if current.shape != loaded.shape:
            assert_broadcastable(current, loaded,
                                 f"Weights are not compatible. Input shapes "
                                 f"were {current.shape} and {loaded.shape}")
            mismatches = np.not_equal(current.shape,
                                      loaded.shape)
            mismatch_idx = np.nonzero(mismatches)[0]
            if len(mismatch_idx) != 1:
                raise NotImplementedError("No loading strategy is "
                                          "implemented for weights "
                                          "with more than 1 "
                                          "mismatching dimension.")
            # Insert the loaded weights in the middle of the new dimension.
            mismatch_idx = mismatch_idx[0]
            length = current.shape[mismatch_idx]
            assert length % 2 == 1, ("The size of the new dimension must be "
                                     "odd.")
            current *= rescale_current
            current = np.swapaxes(current, mismatch_idx, 0)
            loaded = np.swapaxes(loaded, mismatch_idx, 0)
            current[length//2] = loaded[0]
            loaded = np.swapaxes(current, 0, mismatch_idx)
        current_weights[i] = loaded
    return current_weights


def insert_layers(loaded_weights: np.ndarray, current_weights: np.ndarray,
                  rescale_current: float = .1):
    """
    Insert layers from `current_weights` that are missing in `loaded_weights`.

    Insert the missing weights, set the weights on the diagonal and at the
    middle of the new dimension to `1` and set other weights with the newly
    initialized weights, rescaled by `rescale_current`. This is equivalent to
    initializing weights in the lower dimensional setting, plus a noise
    contribution of the newly initialized weights `current_weights`
    off-diagonal.

    This is mostly used for the purpose of adding layers to `RCNN2D.encoder`.
    Therefore, convolutional layers are assumed and the last two dimensions of
    each weight array (filter dimensions) are supposed equal.

    :param loaded_weights: The weights of the recovered checkpoint.
    :type loaded_weights: np.ndarray
    :param current_weights: The current weights of the model, which will be
                            partly overridden.
    :type current_weights: np.ndarray
    :param rescale_current: A scaling factor applied on the newly initialized
                            weights `current_weights`.
    :type rescale_current: float
    """
    current_kernels = current_weights[::2]
    current_biases = current_weights[1::2]
    loaded_kernels = iter(loaded_weights[::2])
    loaded_biases = iter(loaded_weights[1::2])

    loaded_weights = []
    is_loaded_consumed = True
    for current, current_bias in zip(current_kernels, current_biases):
        is_loaded_exhausted = False
        try:
            if is_loaded_consumed:
                loaded = next(loaded_kernels)
                loaded_bias = next(loaded_biases)
            is_loaded_consumed = False
        except StopIteration:
            is_loaded_exhausted = True
        if is_loaded_exhausted or current.shape != loaded.shape:
            current *= rescale_current
            new_dim_idx = np.nonzero(np.array(current.shape) > 1)[0][0]
            length = current.shape[new_dim_idx]
            assert length % 2 == 1, ("The size of the new dimension must "
                                     "be odd.")
            current = np.swapaxes(current, new_dim_idx, 0)
            diagonal = np.identity(current.shape[-1], dtype=bool)
            current[length//2, ..., diagonal] = 1
            current = np.swapaxes(current, 0, new_dim_idx)
        else:
            current = loaded
            current_bias = loaded_bias
            is_loaded_consumed = True
        loaded_weights += [current, current_bias]
    return loaded_weights


def build_encoder(kernels, qties_filters, dilation_rates, input_shape,
                  batch_size, input_dtype=tf.float32, name="encoder"):
    """
    Build the encoder head, a series of CNNs.

    :param kernels: Kernel shapes of each convolution.
    :param qties_filters: Quantity of filters of each CNN.
    :param diltation_rates: Dilation rate in each dimension of each CNN.
    :param input_shape: The shape of the expected input.
    :param batch_size: Quantity of examples in a batch.
    :param input_dtype: Data type of the input.
    :param name: Name of the produced Keras model.

    :return: A Keras model.
    """
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)

    encoder = Sequential(name=name)
    encoder.add(input)
    for kernel, qty_filters, dilation_rate in zip(kernels, qties_filters,
                                                  dilation_rates):
        encoder.add(Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                           padding='same'))
        encoder.add(ReLU())
        encoder.add(Dropout(.5))
    return encoder


def build_rcnn(reps, kernel, qty_filters, dilation_rate, input_shape,
               batch_size, strides=(1, 1, 1), padding='same',
               input_dtype=tf.float32, name="rcnn"):
    """
    Build a RCNN (recurrent convolution neural network).

    :param reps: Quantity of times the CNN is applied.
    :param kernel: Kernel shape of the convolution.
    :param qty_filters: Quantity of filters in the LSTM.
    :param diltation_rate: Dilation rate in each dimension.
    :param input_shape: The shape of the expected input.
    :param batch_size: Quantity of examples in a batch.
    :param input_dtype: Data type of the input.
    :param name: Name of the produced Keras model.

    :return: A Keras model.
    """
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)

    data_stream = input
    conv_3d = Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                     strides=strides, padding=padding)
    activation = ReLU()
    for _ in range(reps):
        data_stream = conv_3d(data_stream)
        data_stream = activation(data_stream)
    rcnn = Model(inputs=input, outputs=data_stream, name=name)
    return rcnn


def build_rnn(units, input_shape, batch_size, input_dtype=tf.float32,
              bidirectional=False, name="rnn"):
    """
    Build a LSTM acting on dimension 1 (the time dimension).

    :param units: Quantity of filters in the LSTM.
    :param input_shape: The shape of the expected input.
    :param batch_size: Quantity of examples in a batch.
    :param input_dtype: Data type of the input.
    :param name: Name of the produced Keras model.

    :return: A Keras model.
    """
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    data_stream = Permute((2, 1, 3))(input)
    batches, shots, timesteps, filter_dim = data_stream.get_shape()
    data_stream = reshape(data_stream,
                          [batches*shots, timesteps, filter_dim])
    lstm = LSTM(units, return_sequences=True)
    if bidirectional:
        lstm = Bidirectional(lstm)
        units *= 2
    data_stream = lstm(data_stream)
    data_stream = reshape(data_stream,
                          [batches, shots, timesteps, units])
    data_stream = Permute((2, 1, 3))(data_stream)

    rnn = Model(inputs=input, outputs=data_stream, name=name)
    return rnn


def assert_broadcastable(arr1: np.ndarray, arr2: np.ndarray,
                         message: str = None):
    """
    Assert that two arrays can be broadcasted together.
    """
    try:
        np.broadcast(arr1, arr2)
    except ValueError:
        if message is None:
            message = (
                f"Arrays are not compatible for broadcasting. Input arrays "
                f"have shapes {arr1.shape} and {arr2.shape}."
            )
        raise AssertionError(message)
