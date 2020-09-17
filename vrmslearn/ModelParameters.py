#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py as h5


class ModelParameters(object):
    """
    This class contains all model parameters needed to generate random models
    and seismic data
    """

    def __init__(self):

        # Number of grid cells in X direction.
        self.NX = 256
        # Number of grid cells in Z direction.
        self.NZ = 256
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = 10.0

        # # Whether free surface is turned on the top face.
        # self.fs = False
        # # Number of padding cells of absorbing boundary.
        # self.Npad = 16
        # # Number of times steps.
        # self.NT = 2048
        # # Time sampling for seismogram (in seconds).
        # self.dt = 0.0009
        # # Peak frequency of input wavelet (in Hertz).
        # self.peak_freq = 10.0
        # # Source wave function selection.
        # self.wavefuns = [1]
        # # Frequency of source peak_freq +- random(df).
        # self.df = 2
        # # Delay of the source.
        # self.tdelay = 2.0 / (self.peak_freq - self.df)
        # # Resampling of the shots time axis.
        # self.resampling = 10
        # # Depth of sources (m).
        # self.source_depth = (self.Npad + 2) * self.dh
        # # Depth of receivers (m).
        # self.receiver_depth = (self.Npad + 2) * self.dh
        # # Receiver interval in grid points.
        # self.dg = 2
        # # Source interval (in 2D).
        # self.ds = 2
        # # Minimum position of receivers (-1 = minimum of grid).
        # self.gmin = None
        # # Maximum position of receivers (-1 = maximum of grid).
        # self.gmax = None
        # self.minoffset = 0
        # # Integer used by SeisCL for pressure source.
        # self.sourcetype = 100

        # # Train on True:  shot gathers, False: CMP.
        # self.train_on_shots = False
        # # The direct arrival is contained in labels.
        # self.identify_direct = True
        # # If true, mute direct arrival.
        # self.mute_dir = False
        # # If true, apply random static to the data.
        # self.mask_firstvel = False
        # self.random_static = False
        # # Maximum static in nb of samples.
        # self.random_static_max = 2
        # # If true, add random noise to the data.
        # self.random_noise = False
        # # Maximum noise relative to data maximum.
        # self.random_noise_max = 0.1
        # # If true, mute random near offset traces.
        # self.mute_nearoffset = False
        # # Maximum offset that can be mutes.
        # self.mute_nearoffset_max = 10
        # # If true, apply a random gain.
        # self.random_time_scaling = False
        #
        # # Model gaussian smoothing.
        # # See smooth_velocity_wavelength in velocity_transformations.
        # # Standard deviation in x.
        # self.model_smooth_x = 0
        # # Standard deviation in t (z smoothing).
        # self.model_smooth_t = 0

        # # Maximum value of vp (in m/s).
        # self.vp_min = 1000.0
        # # Minimum value of vp (in m/s).
        # self.vp_max = 5000.0
        # # Minimum trend for velocity variation in z.
        # self.vp_trend_min = 0
        # # Maximum trend for velocity variation in z.
        # self.vp_trend_max = 0
        # # Maximum velocity difference between 2 layers.
        # self.dvmax = 2000
        #
        # self.rho_var = False
        # # Maximum value of rho.
        # self.rho_min = 2000.0
        # # Minimum value of rho.
        # self.rho_max = 3500.0
        # # Maximum velocity difference between 2 layers.
        # self.drhomax = 800
        #
        # # If true, first layer will be at water velocity.
        # self.marine = False
        # # Minimum velocity of water.
        # self.water_vmin = 1470
        # # Maximum velocity of water.
        # self.water_vmax = 1530
        # # Mean water depth (m).
        # self.water_dmin = 1000
        # # Maximum amplitude of water depth variations.
        # self.water_dmax = 5000
        #
        # # Minimum thickness of a layer (in grid cells).
        # self.layer_dh_min = 50
        # # Minimum number of layers.
        # self.layer_num_min = 5
        # # Fix the number of layers if not 0.
        # self.num_layers = 10
        #
        # # True: 1D, False: 2D model.
        # self.flat = True
        # # If true, first layer angle is 0.
        # self.angle_0 = True
        # # Maximum dip of a layer.
        # self.angle_max = 0
        # # Maximum dip difference between two adjacent layers.
        # self.dangle_max = 5
        # # Max frequency of the layer boundary function.
        # self.max_deform_freq = 0
        # # Min frequency of the layer boundary function.
        # self.min_deform_freq = 0
        # # Maximum amplitude of boundary oscillations.
        # self.amp_max = 25
        # # Maximum nb of frequencies of boundary.
        # self.max_deform_nfreq = 20
        # # Probability that a boundary shape will.
        # self.prob_deform_change = 0.3
        # # Change between two layers.
        # # Add random noise two a layer (% or velocity).
        # self.max_texture = 0
        # # Range of the filter in x for texture creation.
        # self.texture_xrange = 0
        # # Range of the filter in z for texture creation.
        # self.texture_zrange = 0

    def save_parameters_to_disk(self, filename):
        """
        Save all parameters to disk

        @params:
        filename (str) :  name of the file for saving parameters

        @returns:

        """
        with h5.File(filename, 'w') as file:
            for item in self.__dict__:
                file.create_dataset(item, data=self.__dict__[item])

    def read_parameters_from_disk(self, filename):
        """
        Read all parameters from a file

        @params:
        filename (str) :  name of the file containing parameters

        @returns:

        """
        with h5.File(filename, 'r') as file:
            for item in self.__dict__:
                try:
                    self.__dict__[item] = file[item][()]
                except KeyError:
                    pass
