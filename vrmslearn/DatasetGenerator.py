

def generate_dataset(pars: ModelParameters,
                     savepath: str,
                     nexamples: int,
                     seed: int=None,
                     nthread: int=3,
                     workdir: str="./workdir"):
    """
    This method creates a dataset. If multiple threads or processes generate
    the dataset, it may not be totally reproducible due to a different
    random seed attributed to each process or thread.

    @params:
    pars (ModelParameter): A ModelParamter object
    savepath (str)   :     Path in which to create the dataset
    nexamples (int):       Number of examples to generate
    seed (int):            Seed for random model generator
    nthread (int):         Number of processes used to generate examples
    workdir (str):         Name of the directory for temporary files

    @returns:

    """

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    generators = []
    for jj in range(nthread):
        this_workdir = workdir + "_" + str(jj)
        if seed is not None:
            thisseed = seed * (jj + 1)
        else:
            thisseed = seed
        thisgen = DatasetGenerator(pars,
                                   savepath,
                                   this_workdir,
                                   nexamples,
                                   [],
                                   seed=thisseed)
        thisgen.start()
        generators.append(thisgen)
    for gen in generators:
        gen.join()

class DatasetGenerator(Process):
    """
    This class creates a new process to generate seismic data.
    """

    def __init__(self,
                 parameters,
                 savepath: str,
                 workdir: str,
                 nexamples: int,
                 gpus: list,
                 seed: int=None):
        """
        Initialize the DatasetGenerator

        @params:
        parameters (ModelParameter): A ModelParamter object
        savepath (str)   :     Path in which to create the dataset
        workdir (str):         Name of the directory for temporary files
        nexamples (int):       Number of examples to generate
        gpus (list):           List of gpus not to use.
        seed (int):            Seed for random model generator

        @returns:
        """
        super().__init__()

        self.savepath = savepath
        self.workdir = workdir
        self.nexamples = nexamples
        self.parameters = parameters
        self.gpus = gpus
        self.seed = seed
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
        try:
            parameters.save_parameters_to_disk(savepath
                                               + "/model_parameters.hdf5")
        except OSError:
            pass

    def run(self):
        """
        Start the process to generate data
        """
        n = len(fnmatch.filter(os.listdir(self.savepath), 'example_*'))
        gen = SeismicGenerator(model_parameters=self.parameters,
                               gpus=self.gpus)
        if self.seed is not None:
            np.random.seed(self.seed)

        while n < self.nexamples:
            n = len(fnmatch.filter(os.listdir(self.savepath), 'example_*'))
            if self.seed is None:
                np.random.seed(n)
            data, vrms, vp, valid, tlabels = gen.compute_example(self.workdir)
            try:
                gen.write_example(n, self.savepath, data, vrms, vp, valid, tlabels)
                if n % 100 == 0:
                    print("%f of examples computed" % (float(n)/self.nexamples))
            except OSError:
                pass
        if os.path.isdir(self.workdir):
            rmtree(self.workdir)
