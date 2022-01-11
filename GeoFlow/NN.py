# -*- coding: utf-8 -*-
"""
Build the neural network for predicting v_p in 2D and in depth.
"""

import re
from argparse import Namespace
from os import mkdir, listdir
from os.path import join, isdir, isfile, split, exists

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.config import list_physical_devices
from ray.tune.integration.keras import TuneReportCheckpointCallback

from GeoFlow.GeoDataset import GeoDataset

WEIGHTS_NAME = "checkpoint_{epoch}"


class ModelCheckpoint(ModelCheckpoint):
    """Allow saving whole model, and not just weights.

    This workaround is provided by
    github.com/tensorflow/tensorflow/issues/42741#issuecomment-706534711.
    """

    def set_model(self, model):
        self.model = model


class Hyperparameters(Namespace):
    def __init__(self, is_training: bool = True):
        """
        Build the default hyperparameters for `RCNN2D`.

        Mandatory hyperparameters are:
        - `restore_from`: Checkpoint directory from which to restore the model.
                          Defaults to the last checkpoint in `args.logdir`, if
                          `restore_from` is `None`.
        - `epochs`: Quantity of epochs, with `self.steps` iterations per epoch.
        - `step_per_epoch`: Quantity of training iterations per epoch.
        - `batch_size`: Quantity of examples per batch.
        - `seed`: Seed set at the start of training.

        :param is_training: Whether the model is initialized for training or
                            not. This allows building a different network at
                            inference time or dropping the multiple training
                            stages at inference time if using
                            `AutomatedTraining`.
        """
        raise NotImplementedError

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


class NN(Model):
    """
    Define a parent network providing basic model management.
    """
    # Define inputs and outputs names as string in lists stored in class
    # attributes `cls.toinputs` and `cls.tooutputs`.
    toinputs = None
    tooutputs = None

    def __init__(self,
                 input_shapes: dict,
                 params: Hyperparameters,
                 dataset: GeoDataset,
                 checkpoint_dir: str,
                 devices: list = None,
                 run_eagerly: bool = False):
        """
        Build and restore the network.

        :param input_shapes: The shape of every input feature of a single
                             example from the input data.
        :type input_shapes: dict
        :param params: A grouping of hyperparameters.
        :type params: Hyperparameters
        :param dataset: The dataset that is operated on. The dataset can be
                        queried for constants, but fetching data and creating
                        a `tf.data.Dataset` object within a `NN` class is
                        prohibited. Storing a `tf.data.Dataset` object in an
                        attribute causes TensorFlow to register the input
                        pipeline as part of the model, causing the model not to
                        be able to save.
        :type dataset: GeoDataset
        :param checkpoint_dir: The root folder for checkpoints.
        :type checkpoint_dir: str
        :param devices: The list of devices to distribute the model on, such as
                        `[0, 1]`, which will translate to `['/gpu:0',
                        '/gpu:1']`. Defaults to using all GPUs.
        :type devices: list
        :param run_eagerly: Whether to run the model in eager mode or not.
        :type run_eagerly: bool
        """
        if devices:
            all_gpus = list_physical_devices('GPU')
            devices = [':'.join(gpu.name.split(':')[-2:]) for gpu in all_gpus
                       if int(gpu.name.split(':')[-1]) in devices]
        strategy = tf.distribute.MirroredStrategy(devices)
        with strategy.scope():
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.checkpoint_dir = checkpoint_dir
            tf.random.set_seed(self.params.seed)
            inputs = self.build_inputs(input_shapes)
            self._set_inputs(inputs)
            self.build_network(inputs)
            self.setup(run_eagerly)
            self.initialize(input_shapes)
            self.current_epoch = self.restore(self.params.restore_from)

    @property
    def name(self):
        if hasattr(self, "__name"):
            return self.__name
        else:
            return type(self).__name__

    @name.setter
    def name(self, name):
        self.__name = name

    def build_inputs(self, inputs_shape: dict):
        """
        Build input layers.

        Use `tensorflow.keras.layers.Input` layers to define the inputs to the
        model. The filename must be provided through an input layer, even
        though it is not used in the network. Refer to
        `GeoFlow.DefinedNN.Autoencoder.Autoencoder.build_inputs` for an
        example.

        :param input_shapes: The shape of every input feature of a single
                             example from the input data.
        :type input_shapes: dict

        :return: A dictionary of inputs' name-layer pairs.
        """
        inputs = {}
        for name, input_shape in inputs_shape.items():
            input_layer = Input(shape=input_shape,
                                batch_size=self.params.batch_size,
                                dtype=tf.float32)
            inputs[name] = input_layer
        filename_layer = Input(shape=[1],
                               batch_size=self.params.batch_size,
                               dtype=tf.string)
        inputs['filename'] = filename_layer
        return inputs

    def build_network(self, inputs: dict):
        """
        Build the subnetworks of the model.

        Initialize all layers and subnetworks used by the network and store
        them in attributes. Inputs will be fed to layers and subnetworks in
        the `call` method. Use `inputs` to infer the input shapes of the layers
        and subnetworks, if needed.

        :param inputs: The inputs' name-layer pairs.
        :type inputs: dict
        """
        raise NotImplementedError

    def call(self, inputs: dict):
        """
        Apply the neural network to an input tensor.

        Feed `inputs` through the layers and subnetworks. This method is
        required for subclassing `tf.keras.Model`.

        :param inputs: The inputs' name-layer pairs.
        :type inputs: dict

        :return: {out: outputs[out] for out in self.tooutputs}
        """
        raise NotImplementedError

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
            path = path.rstrip("/\\")
            _, filename = split(path)
            current_epoch = filename.split("_")[-1].split(".")[0]
            current_epoch = int(current_epoch)
            self.load_weights(path)
        else:
            current_epoch = 0
        return current_epoch

    def setup(self, run_eagerly: bool = False):
        """
        Setup `Model` prior to fitting.

        Initialize losses, losses' weights and the optimizer. Compile the
        model.

        :param run_eagerly: Whether to run the model in eager mode or not.
                            This parameter can be fed to `self.compile`.
        :type run_eagerly: bool
        """
        raise NotImplementedError

    def initialize(self, input_shapes: list):
        """
        Build the model based on input shapes received.

        This method exists for users who want to call model.build() in a
        standalone way, as a substitute for calling the model on real data to
        build it. The parent `Model.build` method does not support dictionary
        input shapes.

        :param input_shapes: The shape of every input feature of a single
                             example from the input data.
        :type input_shapes: dict
        """
        input_shapes = {name: (None,) + shape
                        for name, shape in input_shapes.items()}
        self.compute_output_shape(input_shapes)

    def load_weights(self, filepath: str, by_name: bool = True,
                     skip_mismatch: bool = False):
        """
        Load weights into the model.

        :param filepath: String, path to the weights file to load. For weight
                         files in TensorFlow format, this is the file prefix
                         (the same as was passed to save_weights).
        :param by_name: `by_name=True` is the only implemented behavior in this
                        subclassed model.
        :param skip_mismatch: This is not implemented.
        """
        if skip_mismatch or not by_name:
            raise NotImplementedError

        loaded_model = tf.keras.models.load_model(filepath, compile=False)
        for loaded_layer in loaded_model.layers:
            name = loaded_layer.name
            current_layer = self.get_layer(name)
            loaded_weights = loaded_layer.get_weights()
            current_layer.set_weights(loaded_weights)

    def launch_training(self, tfdataset, tfvalidate=None,
                        use_tune: bool = False):
        """
        Fit the model to the dataset.

        :param tfdataset: A TensorFlow `Dataset` object created from
                          `GeoFlow.GeoDataset.tfdataset`. `tfdataset` must
                          output pairs of data and labels, but labels are
                          ignored at inference time. `tfdataset` contains the
                          training data.
        :type tfdataset: tf.data.Dataset
        :param tfvalidate: A TensorFlow `Dataset` object created from
                          `GeoFlow.GeoDataset.tfdataset`. `tfvalidate` must
                          output pairs of data and labels, but labels are
                          ignored at inference time. `tfvalidate` contains the
                          validation data.
        :type tfvalidate: tf.data.Dataset
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
        self.fit(tfdataset,
                 validation_data=tfvalidate,
                 epochs=epochs,
                 callbacks=callbacks,
                 initial_epoch=self.current_epoch,
                 steps_per_epoch=self.params.steps_per_epoch,
                 max_queue_size=10,
                 use_multiprocessing=False)

    def build_losses(self):
        """
        Initialize the losses used for training.

        :return: return losses, losses_weights
        """
        raise NotImplementedError

    def launch_testing(self, tfdataset: tf.data.Dataset, savedir: str = None):
        """
        Test the model on a dataset.

        Predictions are saved to a subfolder that has the name of the network
        within the subdataset's directory.

        :param tfdataset: A TensorFlow `Dataset` object created from
                          `GeoFlow.GeoDataset.tfdataset`. `tfdataset` must
                          output pairs of data and labels, but labels are
                          ignored at inference time.
        :type tfdataset: tf.data.Dataset
        :param savedir: The name of the subdirectory within the dataset test
                        directory to save predictions in. Defaults to the name
                        of the network class.
        :type savedir: str
        """
        if savedir is None:
            # Save the predictions to a subfolder that has the name of the
            # network.
            savedir = self.name
        savedir = join(self.dataset.datatest, savedir)
        if not isdir(savedir):
            mkdir(savedir)
        if self.dataset.testsize % self.params.batch_size != 0:
            raise ValueError("Your batch size must be a divisor of your "
                             "dataset length.")

        for data, _ in tfdataset:
            evaluated = self.predict(data,
                                     batch_size=self.params.batch_size,
                                     max_queue_size=10,
                                     use_multiprocessing=False)
            for lbl, out in evaluated.items():
                evaluated[lbl] = out[..., 0]

            for i, example in enumerate(data["filename"]):
                try:
                    example = example.numpy().decode("utf-8")
                except AttributeError:
                    example = example[0]
                exampleid = int(example.split("_")[-1])
                example_evaluated = {lbl: out[i]
                                     for lbl, out in evaluated.items()}
                self.dataset.generator.write_predictions(exampleid, savedir,
                                                         example_evaluated)


def find_latest_checkpoint(logdir: str):
    """
    Find the latest checkpoint that matches `"checkpoint_[0-9]*"`.

    :param logdir: The directory in which to search for the checkpoints.
    :type logdir: str
    """
    expr = re.compile(r"checkpoint_[0-9]*")
    checkpoints = []
    if not exists(logdir):
        return None
    for file in listdir(logdir):
        has_checkpoint_format = expr.match(file)
        is_checkpoint = isfile(file)
        is_model = exists(join(logdir, file, "saved_model.pb"))
        if has_checkpoint_format and (is_checkpoint or is_model):
            checkpoints.append(file.split("_")[-1].split(".")[0])
    if checkpoints:
        has_leading_zeros = checkpoints[0][0] == '0'
        if has_leading_zeros:
            checkpoints = sorted(checkpoints)
            restore_from = checkpoints[-1]
        else:
            checkpoints = [int(f) for f in checkpoints]
            restore_from = str(max(checkpoints))
        restore_from = join(logdir, f"checkpoint_{restore_from}")
    else:
        restore_from = None
    return restore_from
