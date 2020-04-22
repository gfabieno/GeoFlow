#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py as h5

class ModelParameters(object):
    """
    This class contains all model parameters needed to generate random models
    and seismic data
    """

    def __init__(self):

        self.NX = 256                      # number of grid cells in X direction
        self.NZ = 256                      # number of grid cells in Z direction
        self.dh = 10.0          # grid spacing in X, Y, Z directions (in meters)
        self.fs = False         # whether free surface is turned on the top face
        self.Npad = 16           # number of padding cells of absorbing boundary
        self.NT = 2048                                   # number of times steps
        self.dt = 0.0009             # time sampling for seismogram (in seconds)
        self.peak_freq = 10.0       # peak frequency of input wavelet (in Hertz)
        self.wavefuns = [1]                     # Source wave function selection
        self.df = 2                # Frequency of source peak_freq +- random(df)
        self.tdelay = 2.0 / (self.peak_freq - self.df)     # delay of the source
        self.resampling = 10                 # Resampling of the shots time axis
        self.source_depth = (self.Npad + 2) * self.dh     # depth of sources (m)
        self.receiver_depth = (self.Npad + 2) * self.dh # depth of receivers (m)
        self.dg = 2                           # Receiver interval in grid points
        self.ds = 2                                    # Source interval (in 2D)
        self.gmin = None  # Minimum position of receivers (-1 = minimum of grid)
        self.gmax = None  # Maximum position of receivers (-1 = maximum of grid)
        self.minoffset = 0
        self.sourcetype = 100       # integer used by SeisCL for pressure source

        self.train_on_shots = False   # Train on True:  shot gathers, False: CMP
        self.identify_direct = True  # The direct arrival is contained in labels
        self.mute_dir = False                     # If true, mute direct arrival
        self.mask_firstvel = False
        self.random_static = False    # If true, apply random static to the data
        self.random_static_max = 2             # Maximum static in nb of samples
        self.random_noise = False        # If true, add random noise to the data
        self.random_noise_max = 0.1     # Maximum noise relative to data maximum
        self.mute_nearoffset = False   # If true, mute random near offset traces
        self.mute_nearoffset_max = 10         # Maximum offset that can be mutes
        self.random_time_scaling = False          # If true, apply a random gain

        # Model gaussian smoothing
        # see smooth_velocity_wavelength in velocity_transformations
        self.model_smooth_x = 0                        # Standard deviation in x
        self.model_smooth_t = 0          # Standard deviation in t (z smoothing)

        self.vp_min = 1000.0                      # maximum value of vp (in m/s)
        self.vp_max = 5000.0                      # minimum value of vp (in m/s)
        self.vp_trend_min = 0        # Minimum trend for velocity variation in z
        self.vp_trend_max = 0        # Maximum trend for velocity variation in z
        self.dvmax = 2000         # Maximum velocity difference between 2 layers

        self.rho_var = False
        self.rho_min = 2000.0                             # maximum value of rho
        self.rho_max = 3500.0                             # minimum value of rho
        self.drhomax = 800        # Maximum velocity difference between 2 layers

        self.marine = False     # if true, first layer will be at water velocity
        self.water_vmin = 1470   # minimum velocity of water
        self.water_vmax = 1530   # maximum velocity of water
        self.water_dmin = 1000                           # mean water depth (m)
        self.water_dmax = 5000   # maximum amplitude of water depth variations

        self.layer_dh_min = 50    # minimum thickness of a layer (in grid cells)
        self.layer_num_min = 5                        # minimum number of layers
        self.num_layers = 10                 # Fix the number of layers if not 0

        self.flat = True                             # True: 1D, False: 2D model
        self.angle_0 = True                    # It true, first layer angle is 0
        self.angle_max = 0                              # Maximum dip of a layer
        self.dangle_max = 5 # Maximum dip difference between two adjacent layers
        self.max_deform_freq = 0  # Max frequency of the layer boundary function
        self.min_deform_freq = 0  # Min frequency of the layer boundary function
        self.amp_max = 25           # Maximum amplitude of boundary oscillations
        self.max_deform_nfreq = 20       # Maximum nb of frequencies of boundary
        self.prob_deform_change = 0.3   # Probability that a boundary shape will
                                                    # change between two lahyers
        self.max_texture = 0   # Add random noise two a layer (% or velocity)
        self.texture_xrange = 0  # Range of the filter in x for texture creation
        self.texture_zrange = 0  # Range of the filter in z for texture creation


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
