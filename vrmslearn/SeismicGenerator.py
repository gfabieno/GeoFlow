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
from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.SeismicUtilities import random_wavelet_generator


class SeismicGenerator(SeisCL):
    """
    Class to generate seismic data with SeisCL and output an example to build
    a seismic dataset for training.
    """

    def __init__(
                self,
                pars=ModelParameters(),
                workdir="workdir",
                gpu=0,
            ):
        """

        @params:
        pars (ModelParameters): Parameters for data creation
        workdir (str): Working directory for SeisCL (must be unique for each
                       SeismicGenerator objects working in parallel)
        gpu (int): The GPU id on which to compute data.
        """
        super().__init__()

        # Remove old working directory an assign a new one.
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
        self.csts['N'] = np.array([pars.NZ, pars.NX])
        self.csts['ND'] = 2
        self.csts['dh'] = pars.dh  # Grid spacing
        self.csts['nab'] = pars.Npad  # Set padding cells
        self.csts['dt'] = pars.dt  # Time step size
        self.csts['NT'] = pars.NT  # Nb of time steps
        self.csts['f0'] = pars.peak_freq  # Source frequency
        self.csts['seisout'] = pars.rectype  # Output pressure
        self.csts['freesurf'] = int(pars.fs)  # Free surface

        # Assign the GPU to SeisCL.
        nouse = np.arange(0, 16)
        nouse = nouse[nouse != gpu]
        self.csts['no_use_GPUs'] = nouse

        # Source and receiver positions.
        if pars.flat:
            # Add just one source in the middle
            sx = np.arange(pars.NX / 2, 1 + pars.NX / 2) * pars.dh
        else:
            # Compute several sources
            l1 = pars.Npad + 1
            l2 = pars.NX - pars.Npad
            sx = np.arange(l1, l2, pars.ds) * pars.dh
        sz = np.full_like(sx, pars.source_depth)
        sid = np.arange(0, sx.shape[0])

        self.src_pos_all = np.stack(
            [
                sx,
                np.zeros_like(sx),
                sz,
                sid,
                np.full_like(sx, pars.sourcetype),
            ],
            axis=0,
        )
        self.resampling = pars.resampling

        # Add receivers
        if pars.gmin:
            gmin = pars.gmin
        else:
            gmin = pars.Npad
        if pars.gmax:
            gmax = pars.gmax
        else:
            gmax = pars.NX - pars.Npad

        gx0 = np.arange(gmin, gmax, pars.dg) * pars.dh
        gx = np.concatenate([gx0 for s in sx], axis=0)
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        gz = np.full_like(gx, pars.receiver_depth)
        gid = np.arange(0, len(gx))

        self.rec_pos_all = np.stack(
            [
                gx,
                np.zeros_like(gx),
                gz,
                gsid,
                gid,
                np.full_like(gx, 2),
                np.zeros_like(gx),
                np.zeros_like(gx),
            ],
            axis=0,
        )

        self.wavelet_generator = random_wavelet_generator(
            pars.NT,
            pars.dt,
            pars.peak_freq,
            pars.df,
            pars.tdelay,
        )

    def compute_data(self, vp, vs, rho):
        """
        This method generates compute the data for a vp, vs and rho model.

        @params:
        workdir (str)   : A string containing the working direction of SeisCL

        @returns:
        vp (numpy.ndarray)  : Vp velocity
        vs (numpy.ndarray)  : Vs velocity
        rho (numpy.ndarray)  : Density

        @returns:
        data (numpy.ndarray):  The modeled data.
        """

        self.src_all = None  # Reset source to generate new random source.
        self.set_forward(
            self.src_pos_all[3, :],
            {'vp': vp, 'vs': vs, 'rho': rho},
            withgrad=False,
        )
        self.execute()
        data = self.read_data()
        if self.csts['seisout'] == 2:
            data = data[0][::self.resampling, :]  # Resample data to reduce space.
        if self.csts['seisout'] == 1:
            data = data[1][::self.resampling, :]  # Resample data to reduce space.

        return data
