#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to generate the labels (seismic data)
"""


import os
from shutil import rmtree

import h5py as h5
import numpy as np
from SeisCL.SeisCL import SeisCL
from vrmslearn.ModelGenerator import ModelGenerator, interval_velocity_time
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicUtilities import random_wavelet_generator



class SeismicGenerator(SeisCL):
    """
    Class to generate seismic data with SeisCL and output an example to build
    a seismic dataset for training.
    """
    def __init__(self,
                 pars=ModelParameters(),
                 workdir="workdir",
                 gpu=0):

        super().__init__()
        self.workdir = workdir
        # Initialize the modeling engine
        self.csts['N'] = np.array([pars.NZ, pars.NX])
        self.csts['ND'] = 2
        self.csts['dh'] = pars.dh  # Grid spacing
        self.csts['nab'] = pars.Npad  # Set padding cells
        self.csts['dt'] = pars.dt  # Time step size
        self.csts['NT'] = pars.NT  # Nb of time steps
        self.csts['f0'] = pars.peak_freq  # Source frequency
        self.csts['seisout'] = 2  # Output pressure
        self.csts['freesurf'] = int(pars.fs)  # Free surface

        nouse = np.arange(0,16)
        nouse = nouse[nouse != gpu]
        self.csts['no_use_GPUs'] = np.array(nouse)

        if pars.flat:
            # Add a source in the middle
            sx = np.arange(pars.NX / 2,
                           1 + pars.NX / 2) * pars.dh
        else:
            if pars.train_on_shots:
                l1 = pars.Npad + 1
                if pars.gmin and pars.gmin < 0:
                    l1 += -pars.gmin
                l2 = pars.NX - pars.Npad
                if pars.gmax and pars.gmax > 0:
                    l2 += -pars.gmax

                sx = np.arange(l1, l2, pars.ds) * pars.dh
            else:
                # We need to compute the true CMP as layers have a slope.
                # We compute one CMP, with multiple shots with 1 receiver
                sx = np.arange(pars.Npad + 1,
                               pars.NX - pars.Npad,
                               pars.dg) * pars.dh
        sz = sx * 0 + pars.source_depth
        sid = np.arange(0, sx.shape[0])

        self.src_pos_all = np.stack([sx,
                                     sx * 0,
                                     sz,
                                     sid,
                                     sx * 0 + pars.sourcetype], axis=0)
        self.resampling = pars.resampling

        # Add receivers
        if pars.flat or pars.train_on_shots:
            if pars.gmin:
                gmin = pars.gmin
            else:
                gmin = -(pars.NX - 2 * pars.Npad) // 2
            if pars.gmax:
                gmax = pars.gmax
            else:
                gmax = (pars.NX - 2 * pars.Npad) // 2

            gx0 = np.arange(gmin, gmax, pars.dg) * pars.dh
            gx = np.concatenate([s + gx0 for s in sx], axis=0)
            gsid = np.concatenate([s + gx0 * 0 for s in sid], axis=0)

        else:
            # One receiver per source, with the middle point at NX/2
            gx = (pars.NX - sx / pars.dh) * pars.dh
            gsid = sid
        gz = gx * 0 + pars.receiver_depth
        gid = np.arange(0, len(gx))

        self.rec_pos_all = np.stack([gx,
                                   gx * 0,
                                   gz,
                                   gsid,
                                   gid,
                                   gx * 0 + 2,
                                   gx * 0,
                                   gx * 0], axis=0)

        self.wavelet_generator = random_wavelet_generator(pars.NT,
                                                          pars.dt,
                                                          pars.peak_freq,
                                                          pars.df,
                                                          pars.tdelay)

    def compute_data(self, vp, vs, rho):
        """
        This method generates one example, which contains the vp model, vrms,
        the seismic data and the valid vrms time samples.

        @params:
        workdir (str)   : A string containing the working direction of SeisCL

        @returns:
        data (numpy.ndarray)  : Contains the modelled seismic data
        vrms (numpy.ndarray)  : numpy array of shape (self.pars.NT, ) with vrms
                                values in meters/sec.
        vp (numpy.ndarray)    : numpy array (self.pars.NZ, self.pars.NX) for vp.
        valid (numpy.ndarray) : numpy array (self.pars.NT, )containing the time
                                samples for which vrms is valid
        tlabels (numpy.ndarray) : numpy array (self.pars.NT, ) containing the
                                  if a sample is a primary reflection (1) or not
        """

        self.src_all = None #reset source to generate new random source
        self.set_forward(self.src_pos_all[3, :],
                         {'vp': vp, 'vs': vs, 'rho': rho},
                         withgrad=False)
        self.execute()
        data = self.read_data()
        data = data[0][::self.resampling, :]

        return data









