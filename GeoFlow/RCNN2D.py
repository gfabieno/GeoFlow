# -*- coding: utf-8 -*-
"""
Build the neural network for predicting v_p in 2D and in depth.
"""

import re
from argparse import Namespace
from os import mkdir, listdir
from os.path import join, isdir, isfile, split, exists

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv3D, Conv2D, LSTM, Permute, Input
from tensorflow.keras.backend import max as reduce_max, reshape
from ray.tune.integration.keras import TuneReportCheckpointCallback

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.Losses import ref_loss, v_compound_loss
from GeoFlow.SeismicUtilities import build_time_to_depth_converter

WEIGHTS_NAME = "checkpoint_{epoch}"


class ModelCheckpoint(ModelCheckpoint):
    """Allow saving whole model, and not just weights.

    This workaround is provided by
    github.com/tensorflow/tensorflow/issues/42741#issuecomment-706534711.
    """

    def set_model(self, model):
        self.model = model


class Hyperparameters(Namespace):
    def __init__(self):
        """
        Build the default hyperparameters for `RCNN2D`.
        """
        super().__init__()

        # Checkpoint directory from which to restore the model. Defaults to the
        # last checkpoint in `args.logdir`.
        self.restore_from = None

        # Quantity of epochs, with `self.steps` iterations per epoch.
        self.epochs = 5
        # Quantity of training iterations per epoch.
        self.steps_per_epoch = 100
        # Quantity of examples per batch.
        self.batch_size = 10

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


class RCNN2D(Model):
    """
    Combine a recursive CNN and LSTMs to predict 2D v_p velocity.
    """
    tooutputs = ["ref", "vrms", "vint", "vdepth"]
    toinputs = ["shotgather"]

    def __init__(self,
                 input_shape: tuple,
                 params: Hyperparameters,
                 dataset: GeoDataset,
                 checkpoint_dir: str,
                 run_eagerly: bool = False):
        """
        Build and restore the network.

        :param input_shape: The shape of a single example from the input data.
        :type input_shape: tuple
        :param params: A grouping of hyperparameters.
        :type params: Hyperparameters
        :param dataset: Constants `vmin`, `vmax`, `dh`, `dt`, `resampling`,
                        `tdelay`, `nz`, `source_depth` and `receiver_depth` of
                        the dataset are used for time to depth conversion.
        :param checkpoint_dir: The root folder for checkpoints.
        :type checkpoint_dir: str
        :param run_eagerly: Whether to run the model in eager mode or not.
        :type run_eagerly: bool
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.checkpoint_dir = checkpoint_dir
            self.inputs = self.build_inputs(input_shape)
            self.build_network(self.inputs)
            self.setup_training(run_eagerly)
            self.current_epoch = self.restore(self.params.restore_from)

    def build_inputs(self, input_shape):
        """
        Build input layers.

        :return: A dictionary of inputs' name-layer pairs.
        """
        shotgather = Input(shape=input_shape,
                           batch_size=self.params.batch_size,
                           dtype=tf.float32)
        filename = Input(shape=[1],
                         batch_size=self.params.batch_size,
                         dtype=tf.string)
        inputs = {"shotgather": shotgather, "filename": filename}
        self._set_inputs(inputs)
        return inputs

    def build_network(self, inputs: dict):
        """
        Build the subnetworks of the model.

        :param inputs: The inputs' name-layer pairs.
        :type inputs: dict
        """
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
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.encoder.trainable = False

        self.rcnn = build_rcnn(reps=7,
                               kernel=params.rcnn_kernel,
                               qty_filters=params.rcnn_filters,
                               dilation_rate=params.rcnn_dilation,
                               input_shape=self.encoder.output_shape,
                               batch_size=batch_size,
                               name="time_rcnn")
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.rcnn.trainable = False

        self.decoder['ref'] = Conv2D(1, params.decode_ref_kernel,
                                     padding='same',
                                     activation='sigmoid',
                                     input_shape=self.rcnn.output_shape,
                                     batch_size=batch_size, name="ref")

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])
        self.rnn['vrms'] = build_rnn(units=200,
                                     input_shape=shape_after_pooling,
                                     batch_size=batch_size,
                                     name="rnn_vrms")
        if params.freeze_to in ['vrms', 'vint', 'vdepth']:
            self.rnn['vrms'].trainable = False

        input_shape = self.rnn['vrms'].output_shape
        if params.use_cnn:
            self.cnn['vrms'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vrms")
            if params.freeze_to in ['vrms', 'vint', 'vdepth']:
                self.cnn['vrms'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vrms'] = Conv2D(1, params.decode_kernel, padding='same',
                                      activation='sigmoid',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="vrms")

        self.rnn['vint'] = build_rnn(units=200,
                                     input_shape=input_shape,
                                     batch_size=batch_size,
                                     name="rnn_vint")
        if params.freeze_to in ['vint', 'vdepth']:
            self.rnn['vint'].trainable = False

        input_shape = self.rnn['vint'].output_shape
        if params.use_cnn:
            self.cnn['vint'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vint")
            if params.freeze_to in ['vint', 'vdepth']:
                self.cnn['vint'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vint'] = Conv2D(1, params.decode_kernel, padding='same',
                                      activation='sigmoid',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="vint")

        vint_shape = input_shape[1:-1] + (1,)
        self.time_to_depth = build_time_to_depth_converter(self.dataset,
                                                           vint_shape,
                                                           batch_size,
                                                           name="vdepth")

    def call(self, inputs: dict):
        """
        Apply the neural network to an input tensor.

        This is required from subclassing `tf.keras.Model`.

        :param inputs: The inputs' name-layer pairs.
        :type inputs: dict
        """
        params = self.params

        outputs = {}

        data_stream = self.encoder(inputs["shotgather"])
        data_stream = self.rcnn(data_stream)
        with tf.name_scope("global_pooling"):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn['vrms'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vrms'](data_stream)

        outputs['vrms'] = self.decoder['vrms'](data_stream)

        data_stream = self.rnn['vint'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vint'](data_stream)

        data_stream = self.rnn['vint'](data_stream)
        outputs['vint'] = self.decoder['vint'](data_stream)
        outputs['vdepth'] = self.time_to_depth(outputs['vint'])

        return {out: outputs[out] for out in self.tooutputs}

    def restore(self, path: str = None):
        """
        Restore a checkpoint of the model.

        :param path: The path of the checkpoint to load. Defaults to the latest
                     checkpoint in `self.checkpoint_dir`.
        :type path: str

        :return: The current epoch number.
        """
        if path is None:
            path = find_latest_checkpoint(self.checkpoint_dir)
        if path is not None:
            _, filename = split(path)
            current_epoch = filename.split("_")[-1].split(".")[0]
            current_epoch = int(current_epoch)
            self.load_weights(path)
        else:
            current_epoch = 0
        return current_epoch

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

    def setup_training(self, run_eagerly: bool = False):
        """
        Setup `Model` prior to fitting.

        :param run_eagerly: Whether to run the model in eager mode or not.
        :type run_eagerly: bool
        """
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

    def launch_training(self, dataset, use_tune: bool = False):
        """
        Fit the model to the dataset.

        :param use_tune: Whether `ray[tune]` is used in training or not. This
                         modifies the way callbacks and checkpoints are logged.
        :param use_tune: bool
        """
        epochs = self.params.epochs + self.current_epoch

        if not use_tune:
            tensorboard = TensorBoard(log_dir=self.checkpoint_dir,
                                      profile_batch=0)
            checkpoints = ModelCheckpoint(join(self.checkpoint_dir,
                                               WEIGHTS_NAME),
                                          save_freq='epoch',
                                          save_weights_only=False)
            callbacks = [tensorboard, checkpoints]
        else:
            tune_report = TuneReportCheckpointCallback(filename='.',
                                                       frequency=1)
            tune_report._checkpoint._cp_count = self.current_epoch + 1
            callbacks = [tune_report]
        self.fit(dataset,
                 epochs=epochs,
                 callbacks=callbacks,
                 initial_epoch=self.current_epoch,
                 steps_per_epoch=self.params.steps_per_epoch,
                 max_queue_size=10,
                 use_multiprocessing=False)

    def build_losses(self):
        """
        Initialize the losses used for training.
        """
        losses, losses_weights = {}, {}
        for lbl in self.tooutputs:
            if lbl == 'ref':
                losses[lbl] = ref_loss()
            else:
                losses[lbl] = v_compound_loss()
            losses_weights[lbl] = self.params.loss_scales[lbl]

        return losses, losses_weights

    def launch_testing(self, dataset):
        """
        Test the model on the current dataset.

        Predictions are saved to a subfolder that has the name of the network
        within the subdataset's directory.
        """
        # Save the predictions to a subfolder that has the name of the network.
        savedir = join(self.dataset.datatest, type(self).__name__)
        if not isdir(savedir):
            mkdir(savedir)
        if self.dataset.testsize % self.params.batch_size != 0:
            raise ValueError("Your batch size must be a divisor of your "
                             "dataset length.")

        for data, _ in dataset:
            evaluated = self.predict(data,
                                     batch_size=self.params.batch_size,
                                     max_queue_size=10,
                                     use_multiprocessing=False)
            for lbl, out in evaluated.items():
                evaluated[lbl] = out[..., 0]

            for i, example in enumerate(data["filename"]):
                example = example.numpy().decode("utf-8")
                exampleid = int(example.split("_")[-1])
                example_evaluated = {lbl: out[i]
                                     for lbl, out in evaluated.items()}
                self.dataset.generator.write_predictions(exampleid, savedir,
                                                         example_evaluated)


def broadcast_weights(loaded_weights: np.ndarray, current_weights: np.ndarray):
    """
    Broadcast `loaded_weights` on `current_weights`.

    Extend the loaded weights along one missing dimension and rescale the
    weights to account for the duplication. This is equivalent to using an
    average in the missing dimension.

    :param loaded_weights: The weights of the recovered checkpoint.
    :type loaded_weights: np.ndarray
    :param current_weights: The current weights of the model, which will be
                            overridden.
    :type current_weights: np.ndarray
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
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)

    data_stream = input
    conv_3d = Conv3D(qty_filters, kernel, dilation_rate=dilation_rate,
                     padding='same')
    for _ in range(reps):
        data_stream = conv_3d(data_stream)
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
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
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


def assert_broadcastable(arr1: np.ndarray, arr2: np.ndarray,
                         message: str = None):
    """
    Assert that two arrays can be broadcasted together.
    """
    try:
        np.broadcast(arr1, arr2)
    except ValueError:
        if message is None:
            message = "Arrays are not compatible for broadcasting."
        raise AssertionError(message)


def find_latest_checkpoint(logdir: str):
    """
    Find the latest checkpoint that matches `"checkpoint_[0-9]*"`.

    :param logdir: The directory in which to search for the checkpoints.
    :type logdir: str
    """
    expr = re.compile(r"checkpoint_[0-9]*")
    checkpoints = []
    for file in listdir(logdir):
        has_checkpoint_format = expr.match(file)
        is_checkpoint = isfile(file)
        is_model = exists(join(logdir, file, "saved_model.pb"))
        if has_checkpoint_format and (is_checkpoint or is_model):
            checkpoints.append(file.split("_")[-1].split(".")[0])
    checkpoints = [int(f) for f in checkpoints]
    if checkpoints:
        restore_from = str(max(checkpoints))
        restore_from = join(logdir, f"checkpoint_{restore_from}")
    else:
        restore_from = None
    return restore_from
