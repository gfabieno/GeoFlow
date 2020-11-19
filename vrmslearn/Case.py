"""
The case class define the basic class to build a case containing parameters
for creating the dataset and training the neural network.
See Case2Dtest for usage example.
"""

import os
import gc
import fnmatch
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from vrmslearn.DatasetGenerator import DatasetGenerator
from vrmslearn.SeismicGenerator import Acquisition
from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.IOGenerator import Reftime, Vrms, Vint, Vdepth, ShotGather
from typing import List
from tensorflow.keras.utils import Sequence


class Case:
    """
    Base class of a Case. Build a specific case by creating a new class from
    this class and changing the model parameters.
    """
    name = "BaseCase"
    basepath = "Datasets"

    # Seed of the 1st model generated. Seeds fo subsequent models are
    # incremented by 1.
    seed0 = 0

    def __init__(self):
        """
        Initiate a Case by setting the training, validation and test sets size.
        """
        self.trainsize = 5 #10000
        self.validatesize = 0
        self.testsize = 0 # 100
        self.model, self.acquire, self.inputs, self.outputs = self.set_case()
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

    def set_case(self):
        """
        A method that defines the parameters of a case.
        Override to set the parameters of a case.

        :return:
            model: A BaseModelGenerator object that generates models
            acquire: An Acquisition objects that set data creation
            inputs: A dict of objects derived from GraphInput that defines the
                    input of the graph {GraphInput.name: GraphInput()}
            outputs: A dict of objects derived from GraphOutput that defines
                     the output of the graph {GraphOutput.name: GraphOutput()}
        """
        self.trainsize = 5 #10000
        self.validatesize = 0
        self.testsize = 0 # 100

        model = BaseModelGenerator()
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
        Search for examples found in the dataset paths
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
        Read an example from a file and apply the preprocessing.

        @params:
        filename (str): If provided, get the example in filename. If None, get
                        an example for a file list provided by phase.
        phase (str): Either "train", "test" or "validate". Get an example from
                     the "phase" dataset.
        shuffle (bool): If True, draws randomly an example, else give examples
                        in order.
        toinputs (list):  List of the name(s) of the input to the network
        tooutputs (list): List of the name(s) of the output of the network

        @returns:
            inputspre (dict) The preprocessed input data {name1: input1}
            labelspre (dict) The preprocessed labels {name1: label1}
            weightspre (dict) The preprocessed weights {name1: weight1}

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
        Get a batch of data and ouputs in a format compatible with
        tf.Keras.Sequence

        :param batch_size: The batch size
        :param toinput: The name of the input variable
        :param phase: The name of the phase, either "train", "test" or "validate"
        :param tooutputs: A list of names of the output variables

        :returns:
            inputs: The inputs, an array of size [batch_size, input_size]
            outputs: The labels, an array of size [batch_size, 2, output_size]
                     The second dimension, first element is the label, and
                     second is the weight
            filenames: A corresponding filenames for each example in the batch
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
            data, labels, weights, fname = self.get_example(phase=phase,
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

        @params:
        filename (str): If provided, get the example in filename. If None, get
                        a random example.
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

    def animated_dataset(self, phase='train', toinputs=None, tooutputs=None):
        """
        Produces an animation of a dataset, showing the input data, and the
        different labels for each example.

        @params:
        phase (str): Which dataset: either train, test or validate
        """

        fig, axs, ims = self.plot_example(toinputs=toinputs,
                                          tooutputs=tooutputs)
        plt.tight_layout()

        def init():
            self.plot_example(toinputs=toinputs, tooutputs=tooutputs, ims=ims)
            return ims

        def animate(t):
            self.plot_example(toinputs=toinputs, tooutputs=tooutputs, ims=ims)
            return ims

        _ = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(self.files[phase]),
                                    interval=3000, blit=True, repeat=True)
        plt.show()
        gc.collect()





