"""
Classes and functions that combines a SeismicGenerator and a ModelGenerator
to produce a dataset on multiple GPUs. Used by the Case class (Case.py).
"""
import os
from multiprocessing import Process, Queue

import numpy as np
import h5py as h5
from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator, Acquisition
from vrmslearn.LabelGenerator import LabelGenerator
from multiprocessing import Process, Queue


# TODO change seismic to forward with input a dict, making it agnotistic
class SampleGenerator:
    """
    Class to create one example: 1- generate models 2-simulate the data
    """

    def __init__(self, model: BaseModelGenerator, acquire: Acquisition,
                 label: LabelGenerator, gpu: int = 0):
        """
        Create the ModelGenerator and SeismicGenerator objects from pars.
        @params:
        pars (ModelParameters): Parameters for data and model creation
        gpu  (int): The GPU id to use for data computations

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
        @params:
        seed (int): Seed of the model to generate

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
        This method writes one example in the hdf5 format

        @params:
        exampleid (int):        The example id number
        savedir (str)   :       A string containing the directory in which to
                                save the example
        data (numpy.ndarray)  : Contains the modelled seismic data
        labels (list)  :       List of numpy array containing the labels
        filename (str):      If provided, save the example in filename.

        @returns:
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
        This function creates a dataset on multiple GPUs.

        @params:
        pars (ModelParameter): A ModelParameter object containg the parameters
                               for creating examples.
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
                                label=self.label,
                                acquire=self.acquire, gpu=jj)
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
                 sample_generator: SampleGenerator,
                 seeds: Queue):
        """
        Initialize the DatasetGenerator

        @params:
        savepath (str)   :     Path in which to create the dataset
        sample_generator (SampleGenerator): A SampleGenerator object to create
                                            examples
        seeds (Queue):   A Queue containing the seeds of models to create
        """
        super().__init__()

        self.savepath = savepath
        self.sample_generator = sample_generator
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
                data, labels, weights = self.sample_generator.generate(seed)

                self.sample_generator.write(seed, self.savepath, data, labels,
                                            weights, filename=filename)
