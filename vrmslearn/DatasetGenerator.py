"""
Classes and functions that combines a SeismicGenerator and a ModelGenerator
to produce a dataset on multiple GPUs. Used by the Dataset class (Dataset.py).
"""
import os
from multiprocessing import Process, Queue

import numpy as np
import h5py as h5
from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator, Acquisition
from vrmslearn.GraphIO import GraphOutput, GraphInput
from multiprocessing import Process, Queue
from typing import Dict

class DatasetGenerator:
    """
    Class to generate a complete dataset for geophysics learning.
    """

    def __init__(self, model: BaseModelGenerator, acquire: Acquisition,
                 outputs: Dict[str, GraphOutput], inputs: Dict[str, GraphInput],
                 gpu: int = 0):
        """
        To generate a Dataset, we need several elements:

        :param model: A BaseModelGenerator that can create the earth properties
        :param acquire: An Acquisition object controlling the data creation
        :param outputs: A dict of GraphOutput that generate the labels of the
                        network from the generated data and earth properties
        :param inputs: A dict of GraphInput that generate the inputs of the
                       netowrk from the generated data.
        :param gpu:    The gpu id to use for generating the data.
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
        @params:
        seed (int): Seed of the model to generate

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
        This method writes one example in the hdf5 format

        @params:
        exampleid (int):        The example id number
        savedir (str)   :       A string containing the directory in which to
                                save the example
        inputs (dict)  :       Contains the graph inputs {name: input}
        labels (dict)  :       Contains the graph labels {name: label}
        weights (dict)  :      Contains the label weights {name: weight}
        filename (str):      If provided, save the example in filename.

        @returns:
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
        This function creates a dataset on multiple GPUs.

        @params:
        savepath (str)   :     Path in which to create the dataset
        nexamples (int):       Number of examples to generate
        seed0 (int):           First seed of the first example in the dataset.
                               Seeds are incremented by 1 at each example.
        ngpu (int):            Number of available gpus for data creation

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
    This class creates a new process to generate seismic data.
    """

    def __init__(self,
                 savepath: str,
                 data_generator: DatasetGenerator,
                 seeds: Queue):
        """
        Initialize the DatasetGenerator

        @params:
        savepath (str)   :     Path in which to create the dataset
        data_generator (DatasetGenerator): A DatasetGenerator object to create
                                            examples
        seeds (Queue):   A Queue containing the seeds of models to create
        """
        super().__init__()

        self.savepath = savepath
        self.data_generator = data_generator
        self.seeds = seeds
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the process to generate data
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
