"""
Produce a dataset on multiple GPUs.

Used by the `vrmslearn.Case.Case` class.
"""

import os
from multiprocessing import Process, Queue

import numpy as np
import h5py as h5

from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator, Acquisition
from vrmslearn.LabelGenerator import LabelGenerator


# TODO Change seismic to forward with input a dict, making it agnotistic.
class SampleGenerator:
    """
    Create one example.

    First, generate models. Then, simulate the data.
    """

    def __init__(self, model: BaseModelGenerator, acquire: Acquisition,
                 label: LabelGenerator, gpu: int = 0):
        """
        Create a `SeismicGenerator` object from arguments.

        :param pars: Parameters for data and model creation.
        :type pars: ModelParameters
        :param gpu: The GPU id to use for data computations.
        """
        self.model = model
        self.label = label
        self.acquire = acquire
        self.seismic = SeismicGenerator(acquire, model, gpu=gpu,
                                        workdir="workdir%d" % gpu)
        self.files_list = {}

    def generate(self, seed):
        """
        Generate one example

        :param seed: Seed of the model to generate.
        """
        props, _, _ = self.model.generate_model(seed=seed)
        data = self.seismic.compute_data(props)
        labels, weights = self.label.generate_labels(props)

        return data, labels, weights

    def read(self, filename):
        file = h5.File(filename, "r")
        data = file["data"][:]
        labels = []
        for labelname in self.label.label_names:
            labels.append(file[labelname][:])
        weights = []
        for wname in self.label.weight_names:
            weights.append(file[wname][:])
        file.close()

        return data, labels, weights

    def write(self, exampleid, savedir, data, labels, weights, filename=None):
        """
        Write one example in hdf5 format.

        :param exampleid: The example id number.
        :param savedir: The directory in which to save the example.
        :param data: An array with the modeled seismic data.
        :param labels: A List of arrays containing the labels.
        :param filename: If provided, save the example as filename.
        """
        if filename is None:
            filename = os.path.join(savedir, "example_%d" % exampleid)
        else:
            filename = os.path.join(savedir, filename)

        file = h5.File(filename, "w")
        file["data"] = data
        for ii, label in enumerate(labels):
            file[self.label.label_names[ii]] = label
        for ii, weight in enumerate(weights):
            file[self.label.weight_names[ii]] = weight
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
                                label=self.label,
                                acquire=self.acquire, gpu=jj)
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
                 sample_generator: SampleGenerator,
                 seeds: Queue):
        """
        Initialize a `DatasetGenerator` object.

        :param savepath: Path at which to create the dataset.
        :param sample_generator: A `SampleGenerator` object to create examples.
        :type sample_generator: SampleGenerator
        :param seeds: A Queue containing the seeds of models to generate.
        :type seeds: Queue
        """
        super().__init__()

        self.savepath = savepath
        self.sample_generator = sample_generator
        self.seeds = seeds
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the processes to generate data.
        """
        while not self.seeds.empty():
            try:
                seed = self.seeds.get(timeout=1)
            except Queue.Full:
                break
            filename = "example_%d" % seed
            if not os.path.isfile(os.path.join(self.savepath, filename)):
                data, labels, weights = self.sample_generator.generate(seed)

                self.sample_generator.write(seed, self.savepath, data, labels,
                                            weights, filename=filename)
