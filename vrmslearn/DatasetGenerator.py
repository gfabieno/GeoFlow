
import numpy as np
import os
import h5py as h5
from vrmslearn.ModelGenerator import ModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator
from vrmslearn.ModelParameters import ModelParameters
from multiprocessing import Process, Queue, Event, Value


class SampleGenerator:

    def __init__(self, pars: ModelParameters, gpu: int = 0):

        self.model_gen = ModelGenerator(pars)
        self.data_gen = SeismicGenerator(pars, gpu=gpu,
                                         workdir="workdir%d" % gpu)
        self.files_list = {}

    def generate(self, seed):

        vp, vs, rho = self.model_gen.generate_model(seed=seed)
        vp, valid = self.model_gen.generate_labels()
        data = self.data_gen.compute_data(vp, vs, rho)

        return data, [vp, valid]

    def write(self, exampleid , savedir, data, labels, filename=None):
        """
        This method writes one example in the hdf5 format

        @params:
        exampleid (int):        The example id number
        savedir (str)   :       A string containing the directory in which to
                                save the example
        data (numpy.ndarray)  : Contains the modelled seismic data
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        vp (numpy.ndarray)    : numpy array (self.pars.NZ, self.pars.NX) for vp.
        valid (numpy.ndarray) : numpy array (self.pars.NT, )containing the time
                                samples for which vrms is valid
        tlabels (numpy.ndarray) : numpy array (self.pars.NT, ) containing the
                                  if a sample is a primary reflection (1) or not

        @returns:
        """

        if filename is None:
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            pid = os.getpid()
            filename = savedir + "/example_%d_%d" % (exampleid , pid)
        else:
            filename = savedir + "/" + filename

        file = h5.File(filename, "w")
        file["data"] = data
        for ii, label in enumerate(labels):
            file["label%d" % ii] = label
        file.close()


class DatasetProcess(Process):
    """
    This class creates a new process to generate seismic data.
    """

    def __init__(self,
                 savepath: str,
                 sample_generator: SampleGenerator,
                 example_ids: Queue):
        """
        Initialize the DatasetGenerator

        @params:
        savepath (str)   :     Path in which to create the dataset
        workdir (str):         Name of the directory for temporary files
        nexamples (int):       Number of examples to generate
        gpus (list):           List of gpus not to use.
        seed (int):            Seed for random model generator

        @returns:
        """
        super().__init__()

        self.savepath = savepath
        self.sample_generator = sample_generator
        self.examples_ids = example_ids
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the process to generate data
        """

        while not self.examples_ids.empty():
            try:
                seed = self.examples_ids.get(timeout=1)
            except Queue.Full:
                break
            data, labels = self.sample_generator.generate(seed)
            self.sample_generator.write(seed, self.savepath,
                                                data, labels)

def generate_dataset(pars: ModelParameters,
                     savepath: str,
                     nexamples: int,
                     seed0: int=None,
                     ngpu: int=3):
    """
    This method creates a dataset. If multiple threads or processes generate
    the dataset, it may not be totally reproducible due to a different
    random seed attributed to each process or thread.

    @params:
    pars (ModelParameter): A ModelParameter object
    savepath (str)   :     Path in which to create the dataset
    nexamples (int):       Number of examples to generate
    seed0 (int):           First seed of the first example in the dataset.
    ngpu (int):            Number of available gpus for data creation

    @returns:

    """

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    exampleids = Queue()
    for el in np.arange(seed0, seed0 +nexamples):
        exampleids.put(el)

    generators = []
    for jj in range(ngpu):
        thisgen = DatasetProcess(savepath,
                                 SampleGenerator(pars, gpu=jj),
                                 exampleids)
        thisgen.start()
        generators.append(thisgen)
    for gen in generators:
        gen.join()

def aggregate(examples):
    """
    This method aggregates a batch of examples

    @params:
    batch (lsit):       A list of numpy arrays that contain a list with
                        all elements of example.

    @returns:
    batch (numpy.ndarray): A list of numpy arrays that contains all examples
                             for each element of a batch.

    """
    nel = len(examples[0])
    batch = [[] for _ in range(nel)]
    for ii in range(nel):
        batch[ii] = np.stack([el[ii] for el in examples])
    #data = np.stack([el[0] for el in batch])
    batch[0] = np.expand_dims(batch[0], axis=-1)

    return batch