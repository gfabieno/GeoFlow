#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to generate the labels (seismic data) with SeisCL. Requires SeisCL
python interface.
"""

import shutil
import os
import numpy as np
from SeisCL.SeisCL import SeisCL
from vrmslearn.SeismicUtilities import random_wavelet_generator
from vrmslearn.VelocityModelGenerator import BaseModelGenerator


class Acquisition:
    """
    This class contains all model parameters needed to model seismic data
    """

    def __init__(self, model: BaseModelGenerator):

        self.model = model
        # Whether free surface is turned on the top face.
        self.fs = False
        # Number of padding cells of absorbing boundary.
        self.Npad = 16
        # Number of times steps.
        self.NT = 2048
        # Time sampling for seismogram (in seconds).
        self.dt = 0.0009
        # Peak frequency of input wavelet (in Hertz).
        self.peak_freq = 10.0
        # Source wave function selection.
        self.wavefuns = [1]
        # Frequency of source peak_freq +- random(df).
        self.df = 2
        # Delay of the source.
        self.tdelay = 2.0 / (self.peak_freq - self.df)
        # Resampling of the shots time axis.
        self.resampling = 10
        # Depth of sources (m).
        self.source_depth = (self.Npad + 2) * model.dh
        # Depth of receivers (m).
        self.receiver_depth = (self.Npad + 2) * model.dh
        # Receiver interval in grid points.
        self.dg = 2
        # Source interval (in 2D).
        self.ds = 2
        # Minimum position of receivers (-1 = minimum of grid).
        self.gmin = None
        # Maximum position of receivers (-1 = maximum of grid).
        self.gmax = None
        self.minoffset = 0
        # Integer used by SeisCL for pressure source (100) or force in z (2)
        self.sourcetype = 100
        # Integer used by SeisCL indicating which type of recording (2: pressure
        # 1 velocities)
        self.rectype = 2

        self.singleshot = True

    def set_rec_src(self):
        """
        This methods outputs the src_pos and rec_pos arrays providing the
        sources and receiver positions for SeisCL. Override to change which data
        is modelled if needed.

        :return:
        src_pos, rec_pos (np.Array) Provides the source et receiver arrays
        """
        # Source and receiver positions.
        if self.singleshot:
            # Add just one source in the middle
            sx = np.arange(self.model.NX / 2,
                           1 + self.model.NX / 2) * self.model.dh
        else:
            # Compute several sources
            l1 = self.Npad + 1
            l2 = self.model.NX - self.Npad
            sx = np.arange(l1, l2, self.ds) * self.model.dh
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, sx.shape[0])

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)], axis=0)

        # Add receivers
        if self.gmin:
            gmin = self.gmin
        else:
            gmin = self.Npad
        if self.gmax:
            gmax = self.gmax
        else:
            gmax = self.model.NX - self.Npad

        gx0 = np.arange(gmin, gmax, self.dg) * self.model.dh
        gx = np.concatenate([gx0 for _ in sx], axis=0)
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        gz = np.full_like(gx, self.receiver_depth)
        gid = np.arange(0, len(gx))

        rec_pos = np.stack([gx,
                            np.zeros_like(gx),
                            gz,
                            gsid,
                            gid,
                            np.full_like(gx, 2),
                            np.zeros_like(gx),
                            np.zeros_like(gx)], axis=0,)

        return src_pos, rec_pos

    def source_generator(self):
        return random_wavelet_generator(self.NT, self.dt, self.peak_freq,
                                        self.df, self.tdelay)


class SeismicGenerator(SeisCL):
    """
    Class to generate seismic data with SeisCL and output an example to build
    a seismic dataset for training.
    """

    def __init__(self, acquire: Acquisition, model: BaseModelGenerator,
                 workdir="workdir", gpu=0):
        """

        @params:
        acquire (Acquisition): Parameters for data creation
        model (VelocityModelGenerator): Model generator
        workdir (str): Working directory for SeisCL (must be unique for each
                       SeismicGenerator objects working in parallel)
        gpu (int): The GPU id on which to compute data.
        """
        super().__init__()

        self.acquire = acquire
        self.model = model
        # Remove old working directory and assign a new one.
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(self.workdir)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(workdir)
        except FileNotFoundError:
            pass
        self.workdir = workdir

        # Assign constants for modeling with SeisCL.
        self.csts['N'] = np.array([model.NZ, model.NX])
        self.csts['ND'] = 2
        self.csts['dh'] = model.dh  # Grid spacing
        self.csts['nab'] = acquire.Npad  # Set padding cells
        self.csts['dt'] = acquire.dt  # Time step size
        self.csts['NT'] = acquire.NT  # Nb of time steps
        self.csts['f0'] = acquire.peak_freq  # Source frequency
        self.csts['seisout'] = acquire.rectype  # Output pressure
        self.csts['freesurf'] = int(acquire.fs)  # Free surface

        # Assign the GPU to SeisCL.
        nouse = np.arange(0, 16)
        nouse = nouse[nouse != gpu]
        self.csts['no_use_GPUs'] = nouse

        self.src_pos_all, self.rec_pos_all = acquire.set_rec_src()
        self.resampling = acquire.resampling

        self.wavelet_generator = acquire.source_generator()

    def compute_data(self, props: dict):
        """
        This method generates compute the data a seismic properties in props.

        :param props: A Dict containint {name_of_property: array_of_property}

        :return: An array containing the modeled seismic data
        """

        self.src_all = None  # Reset source to generate new random source.
        self.set_forward(self.src_pos_all[3, :], props, withgrad=False)
        self.execute()
        data = self.read_data()
        data = data[0][::self.resampling, :]  # Resample data to reduce space.

        return data
