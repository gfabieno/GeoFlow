"""
The case class define the basic class to build a case containing parameters
for creating the dataset and training the neural network.
See Case2Dtest for usage example.
"""

import os
import fnmatch
import h5py as h5
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from vrmslearn.DatasetGenerator import generate_dataset, SampleGenerator
from vrmslearn.ModelGenerator import ModelGenerator
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicUtilities import random_noise, random_time_scaling, random_static, mute_direct, mute_nearoffset
from vrmslearn.SeismicUtilities import smooth_velocity_wavelength, sortcmp, vdepth2time
from vrmslearn.SeismicGenerator import SeismicGenerator

class Case:
    """
    Base class of a Case. Build a specific case by creating a new class from
    this class and changing the model parameters.
    """
    name = "BaseCase"
    basepath = "Datasets"
    pars = ModelParameters()
    sample_generator = SampleGenerator(pars)
    example_order = ['input'] + list(sample_generator.model_gen.label_names) \
                   + list(sample_generator.model_gen.weight_names)
    seed0 = 0 #Seed of the 1st model generated. Seeds fo subsequent models are
              #incremented by 1

    def __init__(self, trainsize=1, validatesize=0, testsize=0):
        """
        Generate a gaussian wavelet

        @params:
        trainsize (int): Number of examples in the training set.
        validatesize (int): Number of examples in the validation set.
        testsize (int): Number of examples in the test set.

        @returns:
        """
        # Paths of the test, train and validation dataset
        self.datatrain = os.path.join(self.basepath, self.name, "train")
        self.datavalidate = os.path.join(self.basepath, self.name, "validate")
        self.datatest = os.path.join(self.basepath, self.name, "test")

        self.trainsize = trainsize
        self.validatesize = validatesize
        self.testsize = testsize

        # List of examples found in the dataset paths
        self.files = {}
        self.files["train"] = []
        self.files["validate"] = []
        self.files["test"] = []
        self._getfilelist()

    def _getfilelist(self):
        """
        Search for examples found in the dataset paths
        """
        try:
            files = fnmatch.filter(os.listdir(self.datatrain), 'example_*')
            self.files["train"] = [os.path.join(self.datatrain, f)
                                   for f in files]
        except FileNotFoundError:
            pass
        try:
            files = fnmatch.filter(os.listdir(self.datavalidate), 'example_*')
            self.files["validate"] = [os.path.join(self.datavalidate, f)
                                      for f in files]
        except FileNotFoundError:
            pass
        try:
            files = fnmatch.filter(os.listdir(self.datatest), 'example_*')
            self.files["test"] = [os.path.join(self.datatest, f)
                                  for f in files]
        except FileNotFoundError:
            pass

    def generate_dataset(self, ngpu=1):
        """
        Generate the training, testing and validation datasets with ngpus.
        """
        seed0 = self.seed0
        generate_dataset(self.pars, self.datatrain, self.trainsize,
                         ngpu=ngpu, seed0=seed0)

        seed0 += self.trainsize
        generate_dataset(self.pars, self.datavalidate, self.validatesize,
                         ngpu=ngpu, seed0=seed0)

        seed0 += self.validatesize
        generate_dataset(self.pars, self.datatest, self.testsize,
                         ngpu=ngpu, seed0=seed0)

    def get_example(self, filename=None, phase="train"):
        """
        Provide an example.

        @params:
        filename (str): If provided, get the example in filename. If None, get
                        a random example.
        phase (str): Either "train", "test" or "validate". Get an example from
                     the "phase" dataset.

        @returns:
        (list): A list: First element is the data, the rest are labels.

        """
        if filename is None:
            files = self.files[phase]
            if not files:
                self._getfilelist()
                files = self.files[phase]
            if not files:
                raise FileNotFoundError

            filename = random.choice(files)

        data, labels, weights = self.sample_generator.read(filename)

        data, labels, weights = preprocess(data, labels, weights, self.pars)

        return [data] + labels + weights

    def get_dimensions(self):
        """
        Output the dimension of the data and the labels (first label)
        """
        example = self.get_example()
        return [e.shape for e in example]

    def ex2batch(self, examples):
        """
        Pack a list of examples into a dict with the entry name.
        Transforms examples = [ex0, ex1, ex2, ...]
                  -> batch = {names[0]: [ex0[0], ex1[0], ex2[0]],
                              names[1]: [ex0[1], ex1[1], ex2[1]], ...}
        """
        batch = {name: np.stack([el[ii] for el in examples])
                 for ii, name in enumerate(self.example_order)}
        return batch

    def plot_example(self, filename=None):
        """
        Plot the data and the labels of an example.

        @params:
        filename (str): If provided, get the example in filename. If None, get
                        a random example.
        """
        examples = self.get_example(filename=filename)
        data = examples[0]
        labels = examples[1:]

        plot_one_example(data, labels, self.pars)

    def animated_dataset(self, phase='train'):
        """
        Produces an animation of a dataset, showing the input data, and the
        different labels for each example.

        @params:
        phase (str): Which dataset: either train, test or validate
        """
        toplots = self.get_example(phase=phase)
        toplots = [np.reshape(el, [el.shape[0], -1]) for el in toplots]
        clip = 0.01
        vmax = np.max(toplots[0]) * clip
        vmin = -vmax

        fig, axs = plt.subplots(1, len(toplots), figsize=[16, 8])
        im1 = axs[0].imshow(toplots[0], animated=True, vmin=vmin, vmax=vmax,
                            aspect='auto', cmap=plt.get_cmap('Greys'))
        ims = [im1] + [axs[ii].imshow(toplots[ii], animated=True, vmin=0,
                                      vmax=1, aspect='auto', cmap='inferno')
                       for ii in range(1, len(toplots))]
        for ii, ax in enumerate(axs):
            ax.set_title(self.example_order[ii])
            plt.colorbar(ims[ii], ax=ax, orientation="horizontal",
                         pad=0.05, fraction=0.2)

        plt.tight_layout()

        def init():
            for im, toplot in zip(ims, toplots):
                im.set_array(toplot)
            return ims

        def animate(t):
            toplots = self.get_example(phase=phase)
            toplots = [np.reshape(el, [el.shape[0], -1]) for el in toplots]
            for im, toplot in zip(ims, toplots):
                im.set_array(toplot)
            return ims

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.files[phase]),
                                       interval=3000, blit=True, repeat=True)
        plt.show()


def plot_model(self, seed=None):
        """
        Plot a velocity model for this case.

        @params:
        seed (int): If provided, get the model generated by the random seed.
        """
        gen = ModelGenerator(self.pars)
        vp, vs, rho = gen.generate_model(seed=seed)
        plt.imshow(vp, aspect='auto')
        plt.colorbar()
        plt.show()

class CaseCollection:
    """
    A class to build a case from multiples cases
    TODO Test this Cases collection
    """
    def __init__(self, cases):

        self.cases = cases
        self.files = {}
        self.files["train"] = []
        self.files["validate"] = []
        self.files["test"] = []
        for case in cases:
            self.files["train"].append(case.files["train"])
            self.files["validate"].append(case.files["validate"])
            self.files["test"].append(case.files["test"])

    def generate_dataset(self, ngpu=1):
        for case in self.cases:
            case.generate_dataset(ngpu=ngpu)

    def get_example(self, phase="train"):
        case = random.choice(self.cases)
        return case.get_example(phase=phase)

def postprocess(labels, preds, pars, vproc=False):
    """
    A function to postprocess the predictions.

    @params:
    labels  (dict): A dict containing {labelname: label}
    pars   (ModelParameter): A parameters of this case

    @returns:
    labels (dict):      The preprocessed labels {labelname: processed_label}
    """
    if vproc:
        for el in ['vrms', 'vint', 'vdepth']:
            if el in labels:
                labels[el] = labels[el] * (pars.vp_max - pars.vp_min) + pars.vp_min
            if el in preds:
                preds[el] = preds[el] * (pars.vp_max - pars.vp_min) + pars.vp_min
    if 'ref' in preds:
        preds['ref'] = np.argmax(preds['ref'], axis=2)

    return labels, preds



def preprocess(data, labels, weights, pars):
    """
    A function to preprocess the data when a Case object reads an example from
    file.

    @params:
    data (numpy.array): Data array
    labels  (list): A list of numpy.array containg the labels
    pars   (ModelParameter): A parameters of this case

    @returns:
    data (numpy.array): The preprocessed data
    labels (list):      The preprocessed label list
    """
    vp = labels[-1]

    """______________Adding random noises to the data___________________"""
    if pars.random_time_scaling:
        data = random_time_scaling(data, pars.dt * pars.resampling)
    if pars.mute_dir:
        data = mute_direct(data, vp[0], pars)
    if pars.random_static:
        data = random_static(data, pars.random_static_max)
    if pars.random_noise:
        data = random_noise(data, pars.random_noise_max)
    if pars.mute_nearoffset:
        data = mute_nearoffset(data, pars.mute_nearoffset_max)

    """___________________Resort the data according to CMP___________________"""
    gen = SeismicGenerator(pars)
    if not pars.train_on_shots:
        data, datapos = sortcmp(data, gen.src_pos_all, gen.rec_pos_all)
    else:
        data = np.reshape(data, [data.shape[0],
                                 data.shape[1],
                                 gen.src_pos_all.shape[1]])
        datapos = gen.src_pos_all[0, :]

    """____________________Smooth the velocity model_________________________"""
    if pars.model_smooth_x != 0 or pars.model_smooth_t != 0:
        labels[-1] = smooth_velocity_wavelength(labels[-1], pars.dh,
                                                pars.model_smooth_t,
                                                pars.model_smooth_x)

    """________ Resample labels in x to correspond to data position_________"""
    x = np.arange(0, pars.NX) * pars.dh
    ind1 = np.argmax(x >= datapos[0])
    ind2 = -np.argmax(x[::-1] <= datapos[-1])

    labels = [l[:, ind1:ind2:pars.ds] for l in labels]
    weights = [w[:, ind1:ind2:pars.ds] for w in weights]

    for l in labels:
        if l.shape[-1] != data.shape[-1]:
            raise ValueError("number of x positions in label and "
                             "number cmp mismatch")

    # We can predict velocities under the source and receiver arrays only
    sz = int(pars.source_depth / pars.dh)

    labels[-1] = labels[-1][sz:, :]
    weights[-1] = weights[-1][sz:, :]

    data = np.expand_dims(data, axis=-1)
    # labels = [np.expand_dims(label, axis=-1) for label in labels]
    # weights = [np.expand_dims(weight, axis=-1) for weight in weights]

    return data, labels, weights


def plot_one_example(data, labels, pars):
    """
    A function to plot the data and labels of one example. Used by Case objects.

    @params:
    data (numpy.array): Data array
    labels  (list): A list of numpy.array containg the labels
    pars   (ModelParameter): A parameters of this case

    @returns:
    """

    fig, ax = plt.subplots(1, 1, figsize=[16, 8])

    clip = 0.05
    vmax = np.max(data) * clip
    vmin = -vmax
    data = np.reshape(data, [data.shape[0], -1])
    ax.imshow(data,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')

    ax.set_xlabel("Receiver Index", fontsize=12, fontweight='normal')
    ax.set_ylabel("Time Index," + " dt = "
                     + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax.set_title("Shot Gather", fontsize=16, fontweight='bold')

    plt.show()

    fig, ax = plt.subplots(1, len(labels), figsize=[12, 8])
    ims = [[] for _ in range(len(labels))]
    labels[0] = labels[0] * (pars.vp_max-pars.vp_min) + pars.vp_min
    for ii, label in enumerate(labels):
        ims[ii] = ax[ii].imshow(label, cmap=plt.get_cmap('hot'), aspect='auto')
        ax[ii].set_xlabel("X Cell Index," + " dh = " + str(pars.dh) + " m",
                         fontsize=12, fontweight='normal')
        ax[ii].set_ylabel("Z Cell Index," + " dh = " + str(pars.dh) + " m",
                         fontsize=12, fontweight='normal')
        ax[ii].set_title("Label %d" % ii, fontsize=16, fontweight='bold')
        p = ax[ii].get_position().get_points().flatten()
        #axis_cbar = fig.add_axes([p[0], 0.03, p[2] - p[0], 0.02])
        plt.colorbar(ims[ii], ax=ax[ii])
    plt.tight_layout()
    plt.show()
