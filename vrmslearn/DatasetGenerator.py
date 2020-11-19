"""
Produce a dataset on multiple GPUs.

Used by the `vrmslearn.Dataset.Dataset` class.
"""

import numpy as np
import os
import h5py as h5

from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator, Acquisition
from vrmslearn.GraphIO import GraphOutput, GraphInput
from multiprocessing import Process, Queue
from typing import Dict
from vrmslearn.SeismicUtilities import dispersion_curve

class DatasetGenerator:
    """
    Class to generate a complete dataset for geophysics learning.
    """

    def __init__(self, model: BaseModelGenerator, acquire: Acquisition,
                 outputs: Dict[str, GraphOutput], inputs: Dict[str, GraphInput],
                 gpu: int = 0):
        """
        To generate a `Dataset`, we need several elements:

        :param model: A `BaseModelGenerator` that can create the earth properties
        :param acquire: An Acquisition object controlling the data creation
        :param outputs: A dict of GraphOutput that generate the labels of the
                        network from the generated data and earth properties
        :param inputs: A dict of GraphInput that generate the inputs of the
                       netowrk from the generated data.
        :param gpu:    The GPU id to use for generating the data.
        """
        self.model = model
        self.outputs = outputs
        self.inputs = inputs
        self.acquire = acquire
        self.seismic = SeismicGenerator(acquire, model, gpu=gpu,
                                        workdir="workdir%d" % gpu)

    def new_example(self, seed):
        """
        Generate one example

        :param seed: Seed of the model to generate.
        """
        props, _, _ = self.model.generate_model(seed=seed)
        data = self.seismic.compute_data(props)
        inputs = {key: self.inputs[key].generate(data) for key in self.inputs}
        labels = {}
        weights = {}
        for name in self.outputs:
            label, weight = self.outputs[name].generate(props)
            labels[name] = label
            weights[name] = weight

        return inputs, labels, weights

    def read(self, filename: str):
        """
        Read one example from hdf5 file.

        :param filename: Name of the file.

        :returns:
                inputs: A dict {name: input_data}
                labels: A dict {name: label}
                weights: A dict {name: weight}
        """
        file = h5.File(filename, "r")
        inputs = {key: file[key][:] for key in self.inputs}
        labels = {key: file[key][:] for key in self.outputs}
        weights = {key: file[key+"_w"][:] for key in self.outputs}
        file.close()

        return inputs, labels, weights

    def write(self, exampleid, savedir, inputs, labels, weights, filename=None):
        """
        Write one example in hdf5 format.

        @params:
        :param exampleid: The example id number
        :param savedir The directory in which to save the example.
        :param inputs: Contains the graph inputs {name: input}
        :param labels: Contains the graph labels {name: label}
        :param weights: Contains the label weights {name: weight}
        :param filename: If provided, save the example in filename.
        """
        if filename is None:
            filename = os.path.join(savedir, "example_%d" % exampleid)
        else:
            filename = os.path.join(savedir, filename)

        file = h5.File(filename, "w")
        for name in inputs:
            file[name] = inputs[name]
        for name in labels:
            file[name] = labels[name]
        for name in weights:
            file[name+"_w"] = weights[name]
        file.close()

    def generate_dataset(self,
                         savepath: str,
                         nexamples: int,
                         seed0: int = None,
                         ngpu: int = 3):
        """
        Create a dataset on multiple GPUs.

        :param savepath: Root path of the dataset.
        :param nexamples: Quantity of examples to generate.
        :param seed0: First seed of the first example in the dataset. Seeds are
                      incremented by 1 at each example.
        :param ngpu: Quantity of available GPUs for data creation.

        """
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        exampleids = Queue()
        for el in np.arange(seed0, seed0 + nexamples):
            exampleids.put(el)
        generators = []
        for jj in range(ngpu):
            sg = self.__class__(model=self.model,
                                acquire=self.acquire,
                                inputs=self.inputs,
                                outputs=self.outputs,
                                gpu=jj)
            thisgen = DatasetProcess(savepath, sg, exampleids)
            thisgen.start()
            generators.append(thisgen)
        for gen in generators:
            gen.join()


class DatasetProcess(Process):
    """
    Create a new process to generate seismic data.
    """

    def __init__(self,
                 savepath: str,
                 data_generator: DatasetGenerator,
                 seeds: Queue):
        """
        Initialize a `DatasetGenerator` object.

        :param savepath: Path at which to create the dataset.
        :param data_generator: A `DatasetGenerator` object to create examples.
        :param seeds: A `Queue` containing the seeds of models to generate.
        """
        super().__init__()

        self.savepath = savepath
        self.data_generator = data_generator
        self.seeds = seeds
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the process to generate data.
        """
        while not self.seeds.empty():
            try:
                seed = self.seeds.get(timeout=1)
            except Queue.Full:
                break
            filename = "example_%d" % seed
            if not os.path.isfile(os.path.join(self.savepath, filename)):
                data, labels, weights = self.data_generator.new_example(seed)
                self.data_generator.write(seed, self.savepath, data, labels,
                                          weights, filename=filename)
