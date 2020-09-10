#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Class to generate seismic models and labels for training.
"""

import argparse
import copy

import numpy as np
from scipy.signal import gaussian

from vrmslearn.model_parameters import ModelParameters
from vrmslearn.seismic_utilities import (
    smooth_velocity_wavelength, generate_reflections_ttime,
)
from vrmslearn.seismic_utilities import vdepth2time, calculate_vrms


class ModelGenerator(object):
    """
    Generate a seismic model with the generate_model method and output the
    labels, with generate_labels.
    """

    def __init__(self, model_parameters=ModelParameters()):
        """
        This is the constructor for the class.

        @params:
        model_parameters (ModelParameters)   : A ModelParameter object

        @returns:
        """
        self.pars = model_parameters
        self.vp = None
        self.label_names = ('ref', 'vrms', 'vint', 'vdepth')
        self.weight_names = ['tweight', 'dweight']

    def generate_model(self, seed=None):
        """
        Output the media parameters required for seismic modelling, in this
        case vp, vs and rho.

        @params:

        @returns:
        vp (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vp.
        vs (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vs.
        rho (numpy.ndarray) : numpy array (self.pars.NZ, self.pars.NX) for rho
                              values.
        """

        vp, vs, rho, _, _, _ = generate_random_2Dlayered(self.pars, seed=seed)
        vp = smooth_velocity_wavelength(vp, self.pars.dh, 0, 0)
        self.vp = copy.copy(vp)
        return vp, vs, rho

    def generate_labels(self, vp, vs, rho):
        """
        Output the labels attached to modelling of a particular dataset. In
        this case, we want to predict vp in depth from cmp gathers.

        @params:

        @returns:
        vp (numpy.ndarray)  : numpy containg the normalized vp model.
        valid (numpy.ndarray) : numpy array with 1 before the last reflection,
                                0 afterwards
        """
        vp = self.vp
        vrms = np.zeros((self.pars.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            vrms[:, ii] = calculate_vrms(vp[:, ii],
                                         self.pars.dh,
                                         self.pars.Npad,
                                         self.pars.NT,
                                         self.pars.dt,
                                         self.pars.tdelay,
                                         self.pars.source_depth)

        vrms = vrms[::self.pars.resampling, :]
        refs = np.zeros((self.pars.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            refs[:, ii] = generate_reflections_ttime(vp[:, ii], self.pars)
        refs = refs[::self.pars.resampling, :]

        vint = np.zeros((self.pars.NT, vp.shape[1]))
        z0 = int(self.pars.source_depth / self.pars.dh)
        t = np.arange(0, self.pars.NT, 1) * self.pars.dt
        for ii in range(vp.shape[1]):
            vint[:, ii] = vdepth2time(vp[z0:, ii], self.pars.dh, t,
                                      t0=self.pars.tdelay)
        vint = vint[::self.pars.resampling, :]

        tweights = vrms * 0 + 1
        dweights = 2 * np.cumsum(self.pars.dh / vp, axis=0) + self.pars.tdelay
        dweights = dweights - 2 * np.sum(self.pars.dh / vp[:z0, :], axis=0)
        for ii in range(vp.shape[1]):
            i_t = np.argwhere(refs[:, ii] > 0.1).flatten()[-1]
            tweights[i_t:, ii] = 0
            mask = dweights[:, ii] >= i_t * self.pars.dt * self.pars.resampling
            dweights[mask, ii] = 0
            dweights[dweights[:, ii] != 0, ii] = 1

        # Normalize so the labels are between 0 and 1
        vrms = (vrms-self.pars.vp_min) / (self.pars.vp_max-self.pars.vp_min)
        vint = (vint-self.pars.vp_min) / (self.pars.vp_max-self.pars.vp_min)
        vp = (vp-self.pars.vp_min) / (self.pars.vp_max-self.pars.vp_min)

        labels = [refs, vrms, vint, vp]
        weights = [tweights, dweights]

        return labels, weights


def random_fields(NZ, NX, lz=2, lx=2, corrs=None):
    """
    Created a random model with bandwidth limited noise.

    @params:
    NZ (int): Number of cells in Z
    NX (int): Number of cells in X
    lz (int): High frequency cut-off size in z
    lx (int): High frequency cut-off size in x
    @returns:

    """
    if corrs is None:
        nf = 1
        corrs = [1.0]
    else:
        nf = len(corrs)+1
        corrs = [1.0] + corrs

    noise0 = np.random.random([NZ, NX])
    noise0 = noise0 - np.mean(noise0)
    noises = []
    for ii in range(nf):
        noisei = np.random.random([NZ, NX])
        noisei = noisei - np.mean(noisei)
        noise = corrs[ii] * noise0 + (1.0-corrs[ii]) * noisei
        noise = np.fft.fft2(noise)
        noise[0, :] = 0
        noise[:, 0] = 0
        # noise[-1, :] = 0
        # noise[:, -1] = 0

        iz = lz
        ix = lx
        maskz = gaussian(NZ, iz)**2
        maskz = np.roll(maskz, [int(NZ / 2), 0])
        if ix > 0:
            maskx = gaussian(NX, ix)**2
            maskx = np.roll(maskx, [int(NX / 2), 0])
            noise *= maskx
        noise = noise * np.reshape(maskz, [-1, 1])

        noise = np.real(np.fft.ifft2(noise))
        noise = noise / np.max(noise)
        if lx == 0:
            noise = np.stack([noise[:, 0] for _ in range(NX)], 1)

        noises.append(noise)

    return noises


def random_layers(pars, seed=None):
    """
    Genereate a random sequence of layers with different thicknesses

    :param pars: A ModelParameter object
    :param seed: If provided, fix the random seed

    :return: A list containing the thicknesses of the layers

    """
    if seed is not None:
        np.random.seed(seed)

    # Determine the minimum and maximum number of layers
    nmin = pars.layer_dh_min
    nmax = int(pars.NZ / pars.layer_num_min)
    if nmax < nmin:
        print("warning: maximum number of layers smaller than minimum")
    nlmax = int(pars.NZ / nmin)
    nlmin = int(pars.NZ / nmax)
    if pars.num_layers == 0:
        if nlmin < nlmax:
            n_layers = np.random.choice(range(nlmin, nlmax))
        else:
            n_layers = nmin
    else:
        n_layers = int(np.clip(pars.num_layers, nlmin, nlmax))

    # Generate a random number of layers with random thicknesses
    dh = pars.dh
    top_min = int(pars.source_depth / dh + 2 * pars.layer_dh_min)
    layers = (nmin + np.random.rand(n_layers) * (nmax - nmin)).astype(np.int)
    tops = np.cumsum(layers)
    ntos = np.sum(layers[tops <= top_min])
    if ntos > 0:
        layers = np.concatenate([[ntos], layers[tops > top_min]])

    if pars.marine:
        layers[0] = int(
            pars.water_dmin / pars.dh
            + np.random.rand() * (pars.water_dmax-pars.water_dmin) / pars.dh
        )

    return layers


def random_angles(pars, layers, seed=None):
    """
    Generate a random sequence of mean angles of the layers

    :param pars: A ModelParameter object
    :param layers: A list of layer thicknesses
    :param seed: If provided, fix the random seed
    :return: A list containing the angles of the layers
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random angles for each layer
    n_angles = len(layers)
    angles = np.zeros(layers.shape)
    if not pars. angle_0:
        angles[1] = -pars.angle_max + np.random.rand() * 2 * pars.angle_max
    for ii in range(2, n_angles):
        angles[ii] = (
            angles[ii - 1] + (2.0*np.random.rand()-1.) * pars.dangle_max
        )
        if np.abs(angles[ii]) > pars.angle_max:
            angles[ii] = np.sign(angles[ii]) * pars.angle_max

    return angles


def random_velocities(pars, layers, seed=None):
    """
    Generate random velocities for a list of layers

    :param pars: A ModelParameter object
    :param layers: A list of layer thicknesses
    :param seed: If provided, fix the random seed

    :return: A list containing the velocities of the layers
    """
    if seed is not None:
        np.random.seed(seed)

    # Set random maximum and minimum velocities of the model, within range
    vmax = pars.vp_max * 1.0
    vmin = pars.vp_min * 1.0
    vmax = vmax - pars.vp_trend_min * pars.NZ - pars.max_texture * vmax
    vmin = vmin + pars.max_texture * vmin
    vmax = vmax - np.random.rand() * (vmax - vmin) / 2.0
    vmin = vmin + np.random.rand() * (vmax - vmin) / 2.0

    # Generate a random velocity for each layer.
    vels = vmin + np.random.rand(len(layers)) * (vmax - vmin)
    if pars.marine:
        vels[0] = (
            pars.water_vmin + np.random.rand()*pars.water_vmax-pars.water_vmin
        )

    return vels


def create_deformation(max_deform_freq, min_deform_freq,
                       amp_max, max_deform_nfreq, Nmax):
    """
    Create random deformations of a boundary with random harmonic functions

    :param max_deform_freq: Maximum frequency of the harmonic components
    :param min_deform_freq: Minimum frequency of the harmonic components
    :param amp_max: Maximum amplitude of the deformation
    :param max_deform_nfreq: Number of frequencies
    :param Nmax: Number of points of the boundary

    :return:
    An array containing the deformation function
    """
    x = np.arange(0, Nmax)
    deform = np.zeros(Nmax)
    if amp_max > 0 and max_deform_freq > 0:
        nfreqs = np.random.randint(max_deform_nfreq)
        freqs = (
            np.random.rand(nfreqs) * (max_deform_freq-min_deform_freq)
            + min_deform_freq
        )
        phases = np.random.rand(nfreqs) * np.pi * 2
        amps = np.random.rand(nfreqs)
        for ii in range(nfreqs):
            deform += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

        ddeform = np.max(deform)
        if ddeform > 0:
            deform = deform / ddeform * amp_max * np.random.rand()

    return deform


def random_deformations(layers, max_deform_freq,
                        min_deform_freq,
                        amp_max,
                        max_deform_nfreq,
                        prob_deform_change,
                        NX, seed=None):
    """
    Generate a list of deformations of the layer boundaries

    :param layers:
    :param max_deform_freq: Maximum frequency of the harmonic components
    :param min_deform_freq: Minimum frequency of the harmonic components
    :param amp_max: Maximum amplitude of the deformation
    :param max_deform_nfreq: Number of frequencies
    :param prob_deform_change: A probability that two consecutive layers will
                               have different deformations
    :param NX: Number of points of the boundary
    :param seed: If provided, fix the random seed

    :return: A list with deformations for each layer
    """
    if seed is not None:
        np.random.seed(seed)

    deforms = [[]] * len(layers)

    deforms[0] = create_deformation(max_deform_freq,
                                    min_deform_freq,
                                    amp_max,
                                    max_deform_nfreq, NX)
    deform = deforms[0]
    dmax = np.max(np.abs(deform))
    for ii in range(1, len(layers)):
        if np.random.rand() < prob_deform_change:
            deform += create_deformation(max_deform_freq,
                                         min_deform_freq,
                                         amp_max,
                                         max_deform_nfreq, NX)
        deforms[ii] = deform
        dmaxi = np.max(np.abs(deforms[ii]))
        if dmaxi > dmax:
            dmax = dmaxi
    if dmax > amp_max:
        dmaxnew = np.random.rand() * amp_max
        for ii in range(0, len(layers)):
            deforms[ii] = deforms[ii] / dmax * dmaxnew

    return deforms


def generate_2Dmodels(NX, NZ, dh, layers, angles, deforms, lz, lx, props,
                      dprops=None, minmax=None, corrs=None, trends=None,
                      marine=False):
    """
    Generate a 2D model of a correlated properties from the model elements
    provided and  add random heterogeneities to each layers and properties

    :param NX: Grid size in X
    :param NZ: Grid size in Z
    :param dh: Cell size
    :param layers: A list of layer thicknesses
    :param angles: A list of layer angles
    :param deforms: A list of layer deformation
    :param lz: The coherence length in z of the random heterogeneities
    :param lx: The coherence length in x of the random heterogeneities
    :param props: A list of the properties, with the values for each layer
    :param dprops: Maximum size of the heterogeneities for each property
    :param minmax: Minimum and maximum for each property
    :param corrs: A list with correlations between each property
    :param trends: A list with min and max trend with depth
    :param marine: If true, do not add texture in first layer
    :return: A list of 2D grid of the properties
    """

    # Generate the 2D model, from top layers to bottom
    npar = len(props)
    props2D = [np.zeros([NZ, NX]) + p[0] for p in props]
    layers2d = np.zeros([NZ, NX])
    tops = np.cumsum(layers)
    if dprops is not None:
        textures = random_fields(2 * NZ, 2 * NX, lz=lz, lx=lx, corrs=corrs)
        if not marine:
            for n in range(npar):
                textures[n] = textures[n] / np.max(textures[n]) * dprops[n][0]
                props2D[n] += textures[n][:NZ, :NX]

    for ii in range(0, len(layers) - 1):

        if trends is not None:
            for n in range(npar):
                trend = (
                    trends[n][0]
                    + np.random.rand() * (trends[n][1]-trends[n][0])
                )
        if dprops is not None:
            for n in range(npar):
                textures[n] = (
                    textures[n] / np.max(textures[n]) * dprops[n][ii + 1]
                )

        for jj in range(0, NX):
            # Depth of the layer at location x.
            dz = int(np.tan(angles[ii + 1]/360*2*np.pi) * (jj-NX/2))
            # Add deformation component.
            if deforms is not None:
                dz = int(dz + deforms[ii][jj])
            # Check if the interface is inside the model.
            if 0 < tops[ii] + dz < NZ:
                for n in range(npar):
                    props2D[n][tops[ii] + dz:, jj] = props[n][ii + 1]
                layers2d[tops[ii] + dz:, jj] = ii
                if dprops is not None:
                    for n in range(npar):
                        props2D[n][tops[ii] + dz:, jj] += (
                            textures[n][tops[ii]:NZ - dz, jj]
                        )
                if trends is not None:
                    for n in range(npar):
                        props2D[n][tops[ii] + dz:, jj] += (
                            trend * np.arange(tops[ii] + dz, NZ)
                        )
            elif tops[ii] + dz <= 0:
                for n in range(npar):
                    props2D[n][:, jj] = props[n][ii + 1]
                layers2d[:, jj] = ii
                if dprops is not None:
                    for n in range(npar):
                        props2D[n][:, jj] += textures[n][:NZ, jj]
                if trends is not None:
                    for n in range(npar):
                        props2D[n][:, jj] += trend * np.arange(0, NZ)

    # Output the 2D model
    if minmax is not None:
        for n in range(npar):
            props2D[n][props2D[n] < minmax[n][0]] = minmax[n][0]
            props2D[n][props2D[n] > minmax[n][1]] = minmax[n][1]

    return props2D, layers2d


def generate_random_2Dlayered(pars, seed=None):
    """
    This method generates a random 2D model, with parameters given in pars.
    Important parameters are:
        Model size:
        -pars.NX : Number of grid cells in X
        -pars.NZ : Number of grid cells in Z
        -pars.dh : Cell size in meters

        Number of layers:
        -pars.num_layers : Minimum number of layers contained in the model
        -pars.layer_dh_min : Minimum thickness of a layer (in grid cell)
        -pars.source_depth: Depth in meters of the source. Velocity above the
                            source is kept constant.

        Layers dip
        -pars.angle_max: Maximum dip of a layer in degrees
        -pars.dangle_max: Maximum dip difference between adjacent layers

        Model velocity
        -pars.vp_max: Maximum Vp velocity
        -pars.vp_min: Minimum Vp velocity
        -pars.dvmax: Maximum velocity difference of two adajcent layers

        Marine survey parameters
        -pars.marine: If True, first layer is water
        -pars.velwater: water velocity
        -pars.d_velwater: variance of water velocity
        -pars.water_dmin: Minimum water depth
        -pars.water_dmax: Maximum water depth

        Non planar layers
        pars.max_osci_freq: Maximum spatial frequency (1/m) of an interface
        pars.min_osci_freq: Minimum spatial frequency (1/m) of an interface
        pars.amp_max: Minimum amplitude of the ondulation of the interface
        pars.max_osci_nfreq: Maximum number of frequencies of the interface

        Add texture in layers
        pars.texture_zrange
        pars.texture_xrange
        pars.max_texture

    @params:
    pars (str) : A ModelParameters class containing parameters for model
        creation.
    seed (str) : The seed for the random number generator

    @returns:
    vp, vs, rho, vels, layers, angles
    vp (numpy.ndarray) : An array containing the vp model
    vs (numpy.ndarray) : An array containing the vs model (0 for the moment)
    rho (numpy.ndarray) : An array containing the density model
        (2000 for the moment)
    vels (numpy.ndarray) : 1D array containing the mean velocity of each layer
    layers (numpy.ndarray) : 1D array containing the mean thickness of each
        layer, at the center of the model
    angles (numpy.ndarray) : 1D array containing slope of each layer
    """

    if seed is not None:
        np.random.seed(seed)

    layers = random_layers(pars)
    angles = random_angles(pars, layers)
    vels = random_velocities(pars, layers)
    deforms = random_deformations(
        layers,
        pars.max_deform_freq,
        pars.min_deform_freq,
        pars.amp_max,
        pars.max_deform_nfreq,
        pars.prob_deform_change,
        pars.NX,
    )
    lx = 3
    lz = 1.95 * pars.NZ / 2
    # Create velocity variations
    # TODO create correlated densities and vs
    if pars.max_texture > 0:
        dprops = [[pars.max_texture * v for v in vels]]
    else:
        dprops = None
    if pars.vp_trend_max > 0:
        trends = [[pars.vp_trend_min, pars.vp_trend_max]]
    else:
        trends = None
    minmax = [[pars.vp_min, pars.vp_max]]
    props2D, layers2d = generate_2Dmodels(
        pars.NX,
        pars.NZ,
        pars.dh,
        layers,
        angles,
        deforms,
        lz,
        lx,
        [vels],
        dprops=dprops,
        minmax=minmax,
        trends=trends,
        corrs=None,
        marine=pars.marine,
    )

    # Output the 2D model.
    vp = props2D[0]
    vs = vp * 0
    rho = vp * 0 + 2000

    return vp, vs, rho, vels, layers, angles


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ND",
        type=int,
        default=2,
        help="Dimension of the model to display"
    )
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    pars = ModelParameters()
    pars.layer_dh_min = 20
    pars.num_layers = 0
    pars.marine = True
    pars.water_dmin = 100
    pars.water_dmax = 1000
    pars.vp_trend_min = 0
    pars.vp_trend_max = 2
    if args.ND == 1:
        # Maximum nb of frequencies of boundary
        pars.max_deform_nfreq = 0
        # Probability that a boundary shape will change
        pars.prob_deform_change = 0.7
        pars.angle_max = 0
        pars.max_texture = 0

        pars.num_layers = 0
        pars.layer_num_min = 15
        pars.layer_dh_min = 10
        vp, vs, rho, vels, layers, angles = generate_random_2Dlayered(pars)
        plt.imshow(vp)
        plt.show()
        vp = vp[:, 0]
        vp = vp[int(pars.source_depth / pars.dh):]
        vint = vdepth2time(
            vp,
            pars.dh,
            np.arange(0, pars.NT, 1) * pars.dt,
            t0=pars.tdelay,
        )
        vrms = calculate_vrms(
            vp,
            pars.dh,
            pars.Npad,
            pars.NT,
            pars.dt,
            pars.tdelay,
            pars.source_depth,
        )

        plt.plot(vint)
        plt.plot(vrms)
        plt.show()
    else:
        # Max frequency of the layer boundary function
        pars.max_deform_freq = 0.1
        # Min frequency of the layer boundary function
        pars.min_deform_freq = 0.0001
        # Maximum amplitude of boundary deformations
        pars.amp_max = 26
        # Maximum nb of frequencies of boundary
        pars.max_deform_nfreq = 40
        # Probability that a boundary shape will change
        pars.prob_deform_change = 0.7
        pars.angle_max = 20

        pars.num_layers = 0
        pars.layer_num_min = 5
        pars.layer_dh_min = 10
        pars.NT = 2000
        seed = np.random.randint(0, 10000)
        print(seed)
        gen = ModelGenerator(pars)
        vp, vs, rho = gen.generate_model()
        labels, weights = gen.generate_labels(vp, vs, rho)
        fig, ax = plt.subplots(2, len(labels))
        for ii, label in enumerate(labels):
            print(np.max(label))
            ax[0, ii].imshow(label, aspect='auto')
        for ii, weight in enumerate(weights):
            ax[1, ii].imshow(weight, aspect='auto')
        plt.show()
        print(np.max(vp))
        # for lt in [0]:#[0, 10, 50, 100, 150, 200]:
        #     vdepth = smooth_velocity_wavelength(vp, pars.dh, lt*0.001, lt/25)
        #     plt.imshow(vdepth)
        #     plt.show()
