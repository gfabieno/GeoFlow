# -*- coding: utf-8 -*-
"""
Build the neural network for predicting v_p in 2D and in depth.
"""

import re
from os import mkdir, listdir
from os.path import split, join, basename, isdir

import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, callbacks, optimizers
from tensorflow.keras.layers import (Conv3D, Conv2D, LeakyReLU, LSTM, Permute,
                                     Input)
from tensorflow.keras.backend import (max as reduce_max, sum as reduce_sum,
                                      reshape, cumsum, arange, expand_dims)

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.Losses import ref_loss, v_compound_loss

WEIGHTS_NAME = "{epoch:04d}.ckpt"


class Hyperparameters:
    def __init__(self):
        """
        Build the default hyperparameters for `RCNN2D`.
        """
        # Checkpoint directory from which to restore the model. Defaults to the
        # last checkpoint in `args.logdir`.
        self.restore_from = None

        # Quantity of epochs, with `self.steps` iterations per epoch.
        self.epochs = 5
        # Quantity of training iterations per epoch.
        self.steps_per_epoch = 100
        # Quantity of examples per batch.
        self.batch_size = 50

        # The learning rate.
        self.learning_rate = 8E-4
        # Adam optimizer hyperparameters.
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-5
        # Losses associated with each label.
        self.loss_scales = {'ref': .8, 'vrms': .1, 'vint': .1, 'vdepth': .0}

        # Whether to add noise or not to the data.
        self.add_noise = False

        # A label. Set layers up to the decoder of `freeze_to` to untrainable.
        self.freeze_to = None

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

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


class RCNN2D:
    """
    Combine a recursive CNN and LSTMs to predict 2D v_p velocity.
    """
    tooutputs = ["ref", "vrms", "vint", "vdepth"]
    toinputs = ["shotgather"]

    def __init__(self,
                 dataset: GeoDataset,
                 phase: str,
                 params: Hyperparameters,
                 checkpoint_dir: str):
        """
        Build and restore the network.

        :param params: A grouping of hyperparameters.
        :type : Hyperparameters
        :param batch_size: Quantity of examples in a batch.
        """
        self.dataset = dataset
        self.params = params
        self.checkpoint_dir = checkpoint_dir
        self.phase = phase

        batch_size = self.params.batch_size
        self.tfdataset = self.dataset.tfdataset(phase=self.phase,
                                                tooutputs=self.tooutputs,
                                                toinputs=self.toinputs,
                                                batch_size=batch_size)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.inputs = self.build_inputs()
            self.outputs = self.build_network()
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
            self.get_layer = self.model.get_layer

            self.current_epoch = self.restore(self.params.restore_from)

    def build_inputs(self):
        inputs, _, _, _ = self.dataset.get_example(toinputs=self.toinputs)
        shot_gather = inputs["shotgather"]
        shotgather = Input(shape=shot_gather.shape,
                           batch_size=self.params.batch_size,
                           dtype=tf.float32)
        filename = Input(shape=[1],
                         batch_size=self.params.batch_size,
                         dtype=tf.string)
        return {"shotgather": shotgather, "filename": filename}

    def build_network(self):
        params = self.params
        outputs = {}

        data_stream = expand_dims(self.inputs["shotgather"], axis=-1)

        encoder = build_encoder(kernels=params.encoder_kernels,
                                dilation_rates=params.encoder_dilations,
                                qties_filters=params.encoder_filters)
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            encoder.trainable = False
        data_stream = encoder(data_stream)

        time_rcnn = build_rcnn(reps=7,
                               kernel=params.rcnn_kernel,
                               qty_filters=params.rcnn_filters,
                               dilation_rate=params.rcnn_dilation,
                               input_shape=data_stream.shape,
                               batch_size=params.batch_size,
                               name="time_rcnn")
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            time_rcnn.trainable = False
        data_stream = time_rcnn(data_stream)

        with tf.name_scope("global_pooling"):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)

        conv_2d = Conv2D(2, params.decode_ref_kernel, padding='same',
                         name="ref")
        outputs['ref'] = conv_2d(data_stream)

        rnn_vrms = build_rnn(units=200, input_shape=data_stream.shape,
                             batch_size=params.batch_size, name="rnn_vrms")
        if params.freeze_to in ['vrms', 'vint', 'vdepth']:
            rnn_vrms.trainable = False
        data_stream = rnn_vrms(data_stream)

        if params.use_cnn:
            conv_2d = Conv2D(params.cnn_filters, params.cnn_kernel,
                             dilation_rate=params.cnn_dilation,
                             padding='same',
                             name="cnn_vrms")
            if params.freeze_to in ['vrms', 'vint', 'vdepth']:
                conv_2d.trainable = False
            data_stream = conv_2d(data_stream)

        conv_2d = Conv2D(1, params.decode_kernel, padding='same',
                         name="vrms")
        outputs['vrms'] = conv_2d(data_stream)

        rnn_vint = build_rnn(units=200, input_shape=data_stream.shape,
                             batch_size=params.batch_size, name="rnn_vint")
        if params.freeze_to in ['vint', 'vdepth']:
            rnn_vint.trainable = False
        data_stream = rnn_vint(data_stream)

        if params.use_cnn:
            conv_2d = Conv2D(params.cnn_filters, params.cnn_kernel,
                             dilation_rate=params.cnn_dilation,
                             padding='same',
                             name="cnn_vint")
            if params.freeze_to in ['vint', 'vdepth']:
                conv_2d.trainable = False
            data_stream = conv_2d(data_stream)

        conv_2d = Conv2D(1, params.decode_kernel, padding='same',
                         name="vint")
        outputs['vint'] = conv_2d(data_stream)

        vint = outputs['vint']
        time_to_depth = build_time_to_depth_converter(self.dataset,
                                                      vint.shape[1:],
                                                      params.batch_size,
                                                      name="vdepth")
        vdepth = time_to_depth(vint)
        outputs['vdepth'] = vdepth

        return {out: outputs[out] for out in self.tooutputs}

    def restore(self, path=None):
        if path is None:
            filename = find_latest_checkpoint(path)
        if path is not None:
            filename = split(path)[-1]
            current_epoch = int(filename[:4])
            self.load_weights(filename)
        else:
            current_epoch = 0

        return current_epoch

    def load_weights(self, filepath, by_name=True, skip_mismatch=False):
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

        loaded_model = tf.keras.models.load_model(filepath, compile=False)
        current_layer_names = [l.name for l in self.layers]
        for loaded_layer in loaded_model.layers:
            name = loaded_layer.name
            if name not in current_layer_names:
                print(f"Loading layer {name} skipped.")
                continue
            current_layer = self.get_layer(name)
            current_weights = current_layer.get_weights()
            loaded_weights = loaded_layer.get_weights()
            current_weights = broadcast_weights(loaded_weights,
                                                current_weights)
            current_layer.set_weights(current_weights)

    def launch_training(self):
        losses, losses_weights = self.build_losses()

        optimizer = optimizers.Adam(learning_rate=self.params.learning_rate,
                                    beta_1=self.params.beta_1,
                                    beta_2=self.params.beta_2,
                                    epsilon=self.params.epsilon,
                                    name="Adam")
        self.compile(optimizer=optimizer,
                     loss=losses,
                     loss_weights=losses_weights)

        epochs = self.params.epochs + self.current_epoch

        tensorboard = callbacks.TensorBoard(log_dir=self.checkpoint_dir,
                                            profile_batch=0)
        checkpoints = callbacks.ModelCheckpoint(join(self.checkpoint_dir,
                                                     WEIGHTS_NAME),
                                                save_freq='epoch')
        self.fit(self.tfdataset,
                 epochs=epochs,
                 callbacks=[tensorboard, checkpoints],
                 initial_epoch=self.current_epoch,
                 steps_per_epoch=self.params.steps_per_epoch,
                 max_queue_size=10,
                 use_multiprocessing=False)

    def build_losses(self):
        losses, losses_weights = {}, {}
        for lbl in self.tooutputs:
            if lbl == 'ref':
                losses[lbl] = ref_loss()
            else:
                losses[lbl] = v_compound_loss()
            losses_weights[lbl] = self.params.loss_scales[lbl]

        return losses, losses_weights

    def launch_testing(self):
        savepath = join(self.dataset.datatest, "pred")
        if not isdir(savepath):
            mkdir(savepath)

        for data, _ in self.tfdataset:
            evaluated = self.predict(data,
                                     max_queue_size=10,
                                     use_multiprocessing=False)
            for i, (lbl, out) in enumerate(zip(self.tooutputs, evaluated)):
                if lbl != 'ref':
                    evaluated[i] = out[..., 0]

            for i, example in enumerate(data["filename"]):
                example = example.numpy().decode("utf-8")
                example = join(savepath, basename(example))
                with h5.File(example, "w") as savefile:
                    for j, el in enumerate(self.tooutputs):
                        if el in savefile.keys():
                            del savefile[el]
                        savefile[el] = evaluated[j][i, :]

    def animated_predictions(self):
        raise NotImplementedError


def broadcast_weights(loaded_weights, current_weights):
    """
    Broadcast `loaded_weights` on `current_weights`.

    Extend the loaded weights along one missing dimension and rescale the
    weights to account for the duplication. This is equivalent to using an
    average in the missing dimension.
    """
    for i, (current, loaded) in enumerate(zip(current_weights,
                                              loaded_weights)):
        if current.shape != loaded.shape:
            assert_broadcastable(current, loaded,
                                 "Weights are not compatible.")
            mismatches = np.not_equal(current.shape,
                                      loaded.shape)
            mismatch_idx = np.nonzero(mismatches)[0]
            if len(mismatch_idx) != 1:
                raise NotImplementedError("No loading strategy is "
                                          "implemented for weights "
                                          "with more than 1 "
                                          "mismatching dimension.")
            # Extend `loaded` in the missing dimension and rescale
            # it to account for the addition of duplicated layers.
            mismatch_idx = mismatch_idx[0]
            repeats = current.shape[mismatch_idx]
            loaded = np.repeat(loaded, repeats,
                               axis=mismatch_idx)
            loaded /= repeats
            noise = np.random.uniform(0, np.amax(loaded)*1E-2,
                                      size=loaded.shape)
            loaded += noise
        current_weights[i] = loaded
    return current_weights


def build_encoder(kernels, qties_filters, dilation_rates, name="encoder"):
    """
    Build the encoder head, a series of CNNs.

    :param kernels: Kernel shapes of each convolution.
    :param qties_filters: Quantity of filters of each CNN.
    :param diltation_rates: Dilation rate in each dimension of each CNN.
    :param name: Name of the produced Keras model.

    :return: A Keras model.
    """
    encoder = Sequential(name=name)
    for kernel, qty_filters, dilation_rate in zip(kernels, qties_filters,
                                                  dilation_rates):
        encoder.add(Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                           padding='same'))
        encoder.add(LeakyReLU())
    return encoder


def build_rcnn(reps, kernel, qty_filters, dilation_rate, input_shape,
               batch_size, input_dtype=tf.float32, name="rcnn"):
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
    """
    Build a time to depth conversion model in Keras.

    :param case: Constants `vmin`, `vmax`, `dh`, `dt`, `resampling`,
                 `tdelay`, `nz`, `source_depth` and `receiver_depth` of the
                 case are used.
    :param input_size: The shape of the expected input.
    :param batch_size: Quantity of examples in a batch.
    :param input_dtype: Data type of the input.
    :param name: Name of the produced Keras model.

    :return: A Keras model.
    """
    vmax = case.model.vp_max
    vmin = case.model.vp_min
    dh = case.model.dh
    dt = case.acquire.dt
    resampling = case.acquire.resampling
    tdelay = case.acquire.tdelay
    tdelay = round(tdelay / (dt*resampling))  # Convert to unitless time steps.
    nz = case.model.NZ
    source_depth = case.acquire.source_depth
    max_depth = nz - int(source_depth / dh)

    vint = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    actual_vint = vint*(vmax-vmin) + vmin
    # Convert to unitless quantity of grid cells.
    depth_intervals = actual_vint * dt * resampling / (dh*2)
    paddings = [[0, 0], [1, 0], [0, 0], [0, 0]]
    depth_intervals = tf.pad(depth_intervals, paddings, "CONSTANT")
    depths = cumsum(depth_intervals, axis=1)
    depth_delay = reduce_sum(depth_intervals[:, :tdelay+1], axis=1,
                             keepdims=True)
    depths -= depth_delay
    target_depths = arange(max_depth, dtype=tf.float32)
    vdepth = interp_nearest(x=target_depths, x_ref=depths, y_ref=vint, axis=1)

    time_to_depth_converter = Model(inputs=vint, outputs=vdepth, name=name)
    return time_to_depth_converter


def interp_nearest(x, x_ref, y_ref, axis=0):
    """Perform 1D nearest neighbors interpolation in TensorFlow.

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


def find_latest_checkpoint(logdir):
    expr = re.compile(r"[0-9]{4}\.ckpt")
    checkpoints = [f for f in listdir(logdir) if expr.match(f)]
    if checkpoints:
        checkpoints.sort()
        restore_from = checkpoints[-1]
        restore_from = join(logdir, restore_from)
    else:
        restore_from = None
    return restore_from
