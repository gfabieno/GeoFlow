
import os
import fnmatch
import h5py as h5
import random
import matplotlib.pyplot as plt
import numpy as np
from vrmslearn.DatasetGenerator import generate_dataset
from vrmslearn.ModelGenerator import ModelGenerator
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicUtilities import random_noise, random_time_scaling, random_static, mute_direct, mute_nearoffset
from vrmslearn.SeismicUtilities import smooth_velocity_wavelength

class Case:

    name = "BaseCase"
    basepath = "Datasets"
    pars = ModelParameters()
    seed0 = 0
    datatrain = os.path.join(basepath, name, "train")
    datavalidate = os.path.join(basepath, name, "validate")
    datatest = os.path.join(basepath, name, "test")

    def __init__(self, trainsize=1, validatesize=0, testsize=0):

        self.trainsize = trainsize
        self.validatesize = validatesize
        self.testsize = testsize

        self.files = {}
        self.files["train"] = []
        self.files["validate"] = []
        self.files["test"] = []
        self._getfilelist()

    def _getfilelist(self):
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

        if filename is None:
            files = self.files[phase]
            if not files:
                self._getfilelist()
            if not files:
                raise FileNotFoundError()
            filename = random.choice(files, 1)[0]

        file = h5.File(filename, "r")
        data = file['data'][:]
        dnames = list(file.keys())
        dnames = [name for name in dnames if "label" in name]
        dnames.sort()
        labels = [[] for _ in dnames]
        for ii, name in enumerate(dnames):
            labels[ii] = file[name][:]

        data, labels = preprocess(data, labels)

        return data, labels

    def plot_example(self, filename=None):

        data, labels = self.get_example(filename=filename)
        plot_one_example(data, labels, self.pars)

    def plot_model(self, seed=None):
        gen = ModelGenerator(self.pars)
        vp, vs, rho = gen.generate_model(seed=seed)
        plt.imshow(vp, aspect='auto')
        plt.colorbar()
        plt.show()

class CaseCollection:

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

def preprocess(self, data, labels):

    vp = labels[0]
    valid = labels[1]
    if self.pars.random_time_scaling:
        data = random_time_scaling(data,
                                   self.pars.dt * self.pars.resampling)
    if self.pars.mute_dir:
        data = mute_direct(data, vp[0], self.pars)
    if self.pars.random_static:
        data = random_static(data, self.pars.random_static_max)
    if self.pars.random_noise:
        data = random_noise(data, self.pars.random_noise_max)
    if self.pars.mute_nearoffset:
        data = mute_nearoffset(data, self.pars.mute_nearoffset_max)

    if self.pars.model_smooth_x != 0 or self.pars.model_smooth_t != 0:
        vp = smooth_velocity_wavelength(vp, self.pars.dh,
                                        self.pars.model_smooth_t,
                                        self.pars.model_smooth_x)

    return data, [vp, valid]

def plot_one_example(data, label, pars):

    fig, ax = plt.subplots(1, 2, figsize=[16, 8])

    im1 = ax[0].imshow(label, cmap=plt.get_cmap('hot'),
                       aspect='auto',
                       vmin=0.9 * pars.vp_min, vmax=1.1 * pars.vp_max)
    ax[0].set_xlabel("X Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_ylabel("Z Cell Index," + " dh = " + str(pars.dh) + " m",
                     fontsize=12, fontweight='normal')
    ax[0].set_title("P Interval Velocity", fontsize=16, fontweight='bold')
    p = ax[0].get_position().get_points().flatten()
    axis_cbar = fig.add_axes([p[0], 0.03, p[2] - p[0], 0.02])
    plt.colorbar(im1, cax=axis_cbar, orientation='horizontal')

    clip = 0.1
    vmax = np.max(data) * clip
    vmin = -vmax

    ax[1].imshow(data,
                 interpolation='bilinear',
                 cmap=plt.get_cmap('Greys'),
                 vmin=vmin, vmax=vmax,
                 aspect='auto')

    ax[1].set_xlabel("Receiver Index", fontsize=12, fontweight='normal')
    ax[1].set_ylabel("Time Index," + " dt = "
                     + str(pars.dt * 1000 * pars.resampling) + " ms",
                     fontsize=12, fontweight='normal')
    ax[1].set_title("Shot Gather", fontsize=16, fontweight='bold')

    plt.show()