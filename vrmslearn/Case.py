"""
The case class define the basic class to build a case containing parameters
for creating the dataset and training the neural network.
See Case2Dtest for usage example.
"""

import os
import fnmatch
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from vrmslearn.DatasetGenerator import SampleGenerator
from vrmslearn.SeismicGenerator import Acquisition
from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.LabelGenerator import LabelGenerator

class Case:
    """
    Base class of a Case. Build a specific case by creating a new class from
    this class and changing the model parameters.
    """
    name = "BaseCase"
    basepath = "Datasets"

    Model = BaseModelGenerator
    Acquire = Acquisition
    Label = LabelGenerator
    Sample = SampleGenerator

    # Seed of the 1st model generated. Seeds fo subsequent models are
    # incremented by 1.
    seed0 = 0

    def __init__(self, trainsize=1, validatesize=0, testsize=0):
        """
        Generate a gaussian wavelet

        @params:
        trainsize (int): Number of examples in the training set.
        validatesize (int): Number of examples in the validation set.
        testsize (int): Number of examples in the test set.

        @returns:
        """

        self.model = self.Model()
        self.acquire = self.Acquire(self.model)
        self.label = self.Label(acquire=self.acquire, model=self.model)
        self.sample = self.Sample(model=self.model, acquire=self.acquire,
                                  label=self.label)
        self.example_order = ['input', *self.label.label_names,
                              *self.label.weight_names]
        self.set_case()

        # Paths of the test, train and validation dataset.
        self.datatrain = os.path.join(self.basepath, self.name, "train")
        self.datavalidate = os.path.join(self.basepath, self.name, "validate")
        self.datatest = os.path.join(self.basepath, self.name, "test")

        self.trainsize = trainsize
        self.validatesize = validatesize
        self.testsize = testsize

        # List of examples found in the dataset paths.
        self.files = {}
        self.files["train"] = []
        self.files["validate"] = []
        self.files["test"] = []
        self._getfilelist()


    def set_case(self):
        """
        Override this method to set the parameters of a case
        """
        self.acquire.source_depth = (self.acquire.Npad + 2) * self.model.dh
        self.acquire.receiver_depth = (self.acquire.Npad + 2) * self.model.dh
        self.model.texture_xrange = 3
        self.model.texture_zrange = 1.95 * self.model.NZ / 2

    def _getfilelist(self):
        """
        Search for examples found in the dataset paths
        """
        try:
            files = fnmatch.filter(os.listdir(self.datatrain), 'example_*')
            self.files["train"] = [
                os.path.join(self.datatrain, f) for f in files
            ]
        except FileNotFoundError:
            pass
        try:
            files = fnmatch.filter(os.listdir(self.datavalidate), 'example_*')
            self.files["validate"] = [
                os.path.join(self.datavalidate, f) for f in files
            ]
        except FileNotFoundError:
            pass
        try:
            files = fnmatch.filter(os.listdir(self.datatest), 'example_*')
            self.files["test"] = [
                os.path.join(self.datatest, f) for f in files
            ]
        except FileNotFoundError:
            pass

    def generate_dataset(self, ngpu=1):
        """
        Generate the training, testing and validation datasets with ngpus.
        """
        seed0 = self.seed0
        self.sample.generate_dataset(self.datatrain, self.trainsize, ngpu=ngpu,
                                     seed0=seed0)

        seed0 += self.trainsize
        self.sample.generate_dataset(self.datavalidate,self.validatesize,
                                     ngpu=ngpu, seed0=seed0)

        seed0 += self.validatesize
        self.sample.generate_dataset(self.datatest, self.testsize, ngpu=ngpu,
                                     seed0=seed0)

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

        data, labels, weights = self.sample.read(filename)

        data, labels, weights = self.label.preprocess(data, labels, weights)

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
        batch = {
            name: np.stack([el[ii] for el in examples])
            for ii, name in enumerate(self.example_order)
        }
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
        ax.set_ylabel( f"Time Index, "
                       f"dt = {self.acquire.dt * 1000 * self.acquire.resampling}"
                       f" ms", fontsize=12, fontweight='normal')
        ax.set_title("Shot Gather", fontsize=16, fontweight='bold')

        plt.show()

        fig, ax = plt.subplots(1, len(labels), figsize=[12, 8])
        ims = [[] for _ in range(len(labels))]
        labels[0] = labels[0] * (self.model.vp_max - self.model.vp_min) + self.model.vp_min
        for ii, label in enumerate(labels):
            ims[ii] = ax[ii].imshow(label, cmap=plt.get_cmap('hot'),
                                    aspect='auto')
            ax[ii].set_xlabel(
                f"X Cell Index, dh = {self.model.dh} m",
                fontsize=12,
                fontweight='normal',
            )
            ax[ii].set_ylabel(
                f"Z Cell Index, dh = {self.model.dh} m",
                fontsize=12,
                fontweight='normal',
            )
            ax[ii].set_title(f"Label {ii}", fontsize=16, fontweight='bold')
            _ = ax[ii].get_position().get_points().flatten()
            # axis_cbar = fig.add_axes([p[0], 0.03, p[2] - p[0], 0.02])
            plt.colorbar(ims[ii], ax=ax[ii])

        plt.show()

    def animated_dataset(self, phase='train', show="all"):
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
                                      vmax=1,aspect='auto', cmap='inferno')
                       for ii in range(1, len(toplots))]

        for ii, ax in enumerate(axs):
            ax.set_title(self.example_order[ii])
            plt.colorbar(ims[ii], ax=ax, orientation="horizontal", pad=0.05,
                         fraction=0.2)
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

        _ = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(self.files[phase]),
                                    interval=3000, blit=True, repeat=True)
        plt.show()


# def plot_model(self, seed=None):
#     """
#     Plot a velocity model for this case.
#
#     @params:
#     seed (int): If provided, get the model generated by the random seed.
#     """
#     gen = ModelGenerator(self.pars)
#     vp, vs, rho = gen.generate_model(seed=seed)
#     plt.imshow(vp, aspect='auto')
#     plt.colorbar()
#     plt.show()


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
