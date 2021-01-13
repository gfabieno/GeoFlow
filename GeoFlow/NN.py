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
    def __init__(self):
        """
        Build the default hyperparameters for `RCNN2D`.

        Mandatory hyperparameters are:
        - `restore_from`: Checkpoint directory from which to restore the model.
                          Defaults to the last checkpoint in `args.logdir`, if
                          `restore_from` is `None`.
        - `epochs`: Quantity of epochs, with `self.steps` iterations per epoch.
        - `step_per_epoch`: Quantity of training iterations per epoch.
        - `batch_size`: Quantity of examples per batch.
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
        :param run_eagerly: Whether to run the model in eager mode or not.
        :type run_eagerly: bool
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.checkpoint_dir = checkpoint_dir
            inputs = self.build_inputs(input_shapes)
            self._set_inputs(inputs)
            self.build_network(inputs)
            self.setup(run_eagerly)
            self.current_epoch = self.restore(self.params.restore_from)

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

    def launch_training(self, tfdataset, use_tune: bool = False):
        """
        Fit the model to the dataset.

        :param tfdataset: A TensorFlow `Dataset` object created from
                          `GeoFlow.GeoDataset.tfdataset`. `tfdataset` must
                          output pairs of data and labels, but labels are
                          ignored at inference time.
        :type tfdataset: tf.data.Dataset
        :param use_tune: Whether `ray[tune]` is used in training or not. This
                         modifies the way callbacks and checkpoints are logged.
        :param use_tune: bool
        """
        if use_tune:
            self.current_epoch += 1
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
            callbacks = [tune_report]
        self.fit(tfdataset,
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

    def launch_testing(self, tfdataset: tf.data.Dataset):
        """
        Test the model on a dataset.

        Predictions are saved to a subfolder that has the name of the network
        within the subdataset's directory.

        :param tfdataset: A TensorFlow `Dataset` object created from
                          `GeoFlow.GeoDataset.tfdataset`. `tfdataset` must
                          output pairs of data and labels, but labels are
                          ignored at inference time.
        :type tfdataset: tf.data.Dataset
        """
        # Save the predictions to a subfolder that has the name of the network.
        savedir = join(self.dataset.datatest, type(self).__name__)
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
                if lbl == 'ref':
                    evaluated[lbl] = out[..., 1]
                else:
                    evaluated[lbl] = out[..., 0]

            for i, example in enumerate(data["filename"]):
                example = example.numpy().decode("utf-8")
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
