"""
Define the base class for building a dataset.

The `GeoDataset` class is the main interface to define and build a Geophysical
dataset for deep neural network training.

See `DefinedDataset` for examples on how to use this class.
"""

import os
import gc
import fnmatch
from typing import List

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GeoFlow import DatasetGenerator
from GeoFlow import Acquisition
from GeoFlow import EarthModel
from GeoFlow import Reftime, Vrms, Vint, Vdepth, ShotGather


class GeoDataset:
    """
    Base class of a dataset.

    Define a specific dataset by inheriting from this class and changing the
    model parameters.
    """
    name = "BaseDataset"
    basepath = "Datasets"

    # Seed of the 1st model generated. Seeds for subsequent models are
    # incremented by 1.
    seed0 = 0

    def __init__(self):
        """
        Initialize a GeoDataset.
        """
        self.trainsize = 10000
        self.validatesize = 0
        self.testsize = 100
        (self.model, self.acquire,
         self.inputs, self.outputs) = self.set_dataset()
        self.generator = DatasetGenerator(model=self.model,
                                          acquire=self.acquire,
                                          inputs=self.inputs,
                                          outputs=self.outputs)

        # Paths of the test, train and validation dataset.
        self.datatrain = os.path.join(self.basepath, self.name, "train")
        self.datavalidate = os.path.join(self.basepath, self.name, "validate")
        self.datatest = os.path.join(self.basepath, self.name, "test")

        # List of examples found in the dataset paths.
        self.files = {"train": [], "validate": [], "test": []}
        self.shuffled = None
        self._shapes = None

    def set_dataset(self):
        """
        Define the parameters of a dataset.

        Override this method to set the parameters of a dataset.

        :return:
            model: A `EarthModel` object that generates models.
            acquire: An `Acquisition` object that set data creation.
            inputs: A dictionary of names and `GraphInput` objects that define
                    the inputs of the graph, for instance `{graph_input.name:
                    graph_input}` with `graph_input = GraphInput()`.
            outputs: The parallel to `inputs` for graph outputs.
        """
        self.trainsize = 10000
        self.validatesize = 0
        self.testsize = 100

        model = EarthModel()
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ / 2

        acquire = Acquisition(model=model)
        acquire.source_depth = (acquire.Npad + 2) * model.dh
        acquire.receiver_depth = (acquire.Npad + 2) * model.dh
        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {Reftime.name: Reftime(model=model, acquire=acquire),
                   Vrms.name: Vrms(model=model, acquire=acquire),
                   Vint.name: Vint(model=model, acquire=acquire),
                   Vdepth.name: Vdepth(model=model, acquire=acquire)}

        return model, acquire, inputs, outputs

    def _getfilelist(self, phase=None):
        """
        Search for examples found in the dataset directory.
        """
        phases = {"train": self.datatrain,
                  "validate": self.datavalidate,
                  "test": self.datatest}
        if phase is not None:
            phases = {phase: phases[phase]}

        for el in phases:
            try:
                files = fnmatch.filter(os.listdir(phases[el]), 'example_*')
                self.files[el] = [os.path.join(phases[el], f) for f in files]
            except FileNotFoundError:
                pass

    def generate_dataset(self, ngpu=1):
        """
        Generate the training, testing and validation datasets with ngpus.
        """
        seed0 = self.seed0
        self.generator.generate_dataset(self.datatrain, self.trainsize,
                                        ngpu=ngpu, seed0=seed0)

        seed0 += self.trainsize
        self.generator.generate_dataset(self.datavalidate, self.validatesize,
                                        ngpu=ngpu, seed0=seed0)

        seed0 += self.validatesize
        self.generator.generate_dataset(self.datatest, self.testsize,
                                        ngpu=ngpu, seed0=seed0)

    def get_example(self, filename=None, phase="train", shuffle=True,
                    toinputs=None, tooutputs=None):
        """
        Read an example from a file and apply preprocessing.

        :param filename: If provided, get the example in filename. If None, get
                         a random example for a file list provided by phase.
        :param phase: Either "train", "test" or "validate". Get an example from
                      the "phase" dataset.
        :param shuffle: If True, draws randomly an example, else give examples
                        in order.
        :param toinputs: List of the name(s) of the inputs to the network.
        :param tooutputs: List of the name(s) of the outputs of the network.

        :return:
            inputspre: A dictionary of inputs' name-values pairs.
            labelspre: A dictionary of labels' name-values pairs.
            weightspre: A dictionary of weights' name-values pairs.
            filename: The filename of the example.
        """
        if toinputs is None:
            toinputs = [el for el in self.inputs]
        if tooutputs is None:
            tooutputs = [el for el in self.outputs]

        if filename is None:
            if not self.files[phase] or self.shuffled != shuffle:
                self._getfilelist(phase=phase)
                if not self.files[phase]:
                    raise FileNotFoundError
                if shuffle:
                    np.random.shuffle(self.files[phase])
                self.shuffled = shuffle
            filename = self.files[phase].pop()

        inputs, labels, weights = self.generator.read(filename)
        inputspre = {key: self.inputs[key].preprocess(inputs[key], labels)
                     for key in toinputs}
        labelspre = {}
        weightspre = {}
        for key in tooutputs:
            label, weight = self.outputs[key].preprocess(labels[key],
                                                         weights[key])
            labelspre[key] = label
            weightspre[key] = weight

        return inputspre, labelspre, weightspre, filename

    def get_batch(self,
                  batch_size: int,
                  phase: str,
                  toinput: str = None,
                  tooutputs: List[str] = None):
        """
        Get a batch of data and outputs.

        The format is compatible with `tf.Keras.Sequence`.

        :param batch_size: The batch size.
        :param toinput: The name of the input variable.
        :param phase: The name of the phase. Either `"train"`, `"test"` or
                      `"validate"`.
        :param tooutputs: A list of names of the output variables.

        :returns:
            inputs: The inputs, an array of size `[batch_size, input_size]`.
            outputs: The labels, an array of size `[batch_size, 2,
                     output_size]`. The items of the second dimension are the
                     labels and weights, respectively.
            filenames: A corresponding filename for each example in the batch.
        """
        if toinput is None:
            toinput = list(self.inputs.keys())[0]
        elif type(toinput) is list:
            raise TypeError("get_batch only supports single inputs for now")

        if tooutputs is None:
            tooutputs = [el for el in self.outputs]

        filenames = []
        inputs = None
        outputs = [None for _ in tooutputs]
        for i in range(batch_size):
            (data, labels,
             weights, fname) = self.get_example(phase=phase,
                                                toinputs=[toinput],
                                                tooutputs=tooutputs)
            filenames.append(fname)
            if inputs is None:
                input_size = data[toinput].shape
                inputs = np.empty([batch_size, *input_size])
            inputs[i] = data[toinput]
            for j, name in enumerate(tooutputs):
                if outputs[j] is None:
                    out_shape = labels[name].shape
                    outputs[j] = np.empty([batch_size, 2, *out_shape])
                outputs[j][i] = [labels[name], weights[name]]

        return inputs, outputs, filenames

    def plot_example(self, filename=None, toinputs=None, tooutputs=None,
                     ims=None):
        """
        Plot the data and the labels of an example.

        :param filename: If provided, get the example in filename. If None, get
                         a random example.
        :param toinputs: List of the name(s) of the inputs to the network.
        :param tooutputs: List of the name(s) of the outputs of the network.
        :param ims:      List of return values of plt.imshow to update.
        """

        inputs, labels, weights, _ = self.get_example(filename=filename,
                                                      toinputs=toinputs,
                                                      tooutputs=tooutputs)

        nplot = np.max([len(inputs), len(labels)])
        if ims is None:
            fig, axs = plt.subplots(3, nplot, figsize=[16, 8], squeeze=False)
            ims = [None for _ in range(len(inputs)+len(labels)+len(weights))]
        else:
            fig = None
            axs = np.zeros((3, nplot))

        n = 0
        for ii, name in enumerate(inputs):
            ims[n] = self.inputs[name].plot(inputs[name],
                                            axs[0, ii],
                                            im=ims[n])
            n += 1
        for ii, name in enumerate(labels):
            ims[n] = self.outputs[name].plot(labels[name],
                                             axs[1, ii],
                                             im=ims[n])
            n += 1
        for ii, name in enumerate(weights):
            ims[n] = self.outputs[name].plot(weights[name],
                                             axs[2, ii],
                                             im=ims[n])
            n += 1

        return fig, axs, ims

    def animate(self, phase='train', toinputs=None, tooutputs=None):
        """
        Produce an animation of a dataset.

        Show the input data and the labels for each example.

        :param phase: Which dataset to animate. Either `"train"`, `"test"` or
                      `"validate"`.
        :param toinputs: List of the name(s) of the inputs to the network.
        :param tooutputs: List of the name(s) of the outputs of the network.
        """
        fig, axs, ims = self.plot_example(toinputs=toinputs,
                                          tooutputs=tooutputs)
        plt.tight_layout()

        def init():
            self.plot_example(toinputs=toinputs, tooutputs=tooutputs, ims=ims)
            return ims

        def animate(_):
            self.plot_example(toinputs=toinputs, tooutputs=tooutputs, ims=ims)
            return ims

        _ = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(self.files[phase]),
                                    interval=3000, blit=True, repeat=True)
        plt.show()
        gc.collect()

    def tfdataset(self,
                  phase: str = "train",
                  shuffle: bool = True,
                  tooutputs: List[str] = None,
                  toinputs: List[str] = None,
                  batch_size: int = 1,
                  num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                  ):
        """
        Outputs at tf.data.dataset to feed a tf or keras network.

        :param phase: Either "train", "test" or "validate". Get an example from
                      the "phase" dataset.
        :param tooutputs: The list of the name of the desired outputs.
        :param toinputs: The list of the name of the inputs.
        :param shuffle: If True, shuffles the examples.
        :param batch_size: The size of a batch.
        :param num_parallel_calls: Number of parallel threads for data reading.
                                   By default, determined automatically by tf.

        :return: A tf.dataset object outputting the batch of examples.

        """

        phases = {"train": self.datatrain,
                  "validate": self.datavalidate,
                  "test": self.datatest}
        pathstr = os.path.join(phases[phase], 'example_*')
        tfdataset = tf.data.Dataset.list_files(pathstr, shuffle=shuffle)

        def get_example(fname):
            (data, labels, weights, fname) = self.get_example(filename=fname)
            data = [data[el] for el in toinputs]
            labels = [labels[el] for el in tooutputs]
            weights = [weights[el] for el in tooutputs]
            return data, labels, weights, fname

        def tf_fun(x):
            otype = [tf.float32, tf.float32, tf.float32, tf.string]
            return tf.numpy_function(get_example, inp=[x], Tout=otype)

        tfdataset = tfdataset.map(tf_fun,
                                  num_parallel_calls=num_parallel_calls,
                                  deterministic=False)
        tfdataset = tfdataset.batch(batch_size=batch_size)

        return tfdataset