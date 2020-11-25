# -*- coding: utf-8 -*-
"""
Generate the labels (seismic data) using SeisCL.

Requires the SeisCL python interface.
"""

import os
import shutil

import numpy as np

from SeisCL.SeisCL import SeisCL
from vrmslearn.SeismicUtilities import random_wavelet_generator
from vrmslearn.VelocityModelGenerator import (BaseModelGenerator,
                                              MaswModelGenerator)
import tensorflow as tf


class Acquisition:
    """
    Define all model parameters needed to model seismic data.
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
        # Integer used by SeisCL for pressure source (100) or force in z (2).
        self.sourcetype = 100
        # Integer used by SeisCL indicating which type of recording. Either
        # 2) pressure or 1) velocities.
        self.rectype = 2

        self.singleshot = True

    def set_rec_src(self):
        """
        Provide the sources' and receivers' positions for SeisCL.

        Override to change which data is modelled if needed.

        :return:
            src_pos: Source array.
            rec_pos: Receiver array.
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


class Aquisition_masw(Acquisition):
    def set_rec_src(self):

        dg = self.dg
        ng = 24  # Quantity of geophones.

        if dg == 'all':
            spacing = [3, 1, 0.5]
            src_pos = []
            rec_pos = []

            for s in spacing:
                gmin = self.Npad + 20*self.model.dh
                gmax = gmin + ng * s
                if s == 3:
                    sx = [gmin - 20, gmin - 5, gmax + 5, gmax + 20]
                    sid = np.arange(0, 4)
                elif s == 1:
                    sx = [gmin - 10, gmin - 3, gmax + 3, gmax + 10]
                    sid = np.arange(4, 8)
                elif s == 0.5:
                    sx = [gmin - 5, gmax + 5]
                    sid = np.arange(8, 10)

                sz = np.full_like(sx, self.source_depth)
                Src_pos = np.stack([sx,
                                    np.zeros_like(sx),
                                    sz,
                                    sid,
                                    np.full_like(sx, self.sourcetype)],
                                   axis=0)

                gx0 = np.arange(gmin, gmax, s) * self.model.dh
                gx = np.concatenate([gx0 for _ in sx], axis=0)
                gsid = np.concatenate([np.full_like(gx0, s) for s in sid],
                                      axis=0)
                gz = np.full_like(gx, self.receiver_depth)
                gid = np.arange(0, len(gx))

                Rec_pos = np.stack([gx,
                                    np.zeros_like(gx),
                                    gz,
                                    gsid,
                                    gid,
                                    np.full_like(gx, 2),
                                    np.zeros_like(gx),
                                    np.zeros_like(gx)], axis=0)

                src_pos.append(Src_pos)
                rec_pos.append(Rec_pos)
            src_pos = np.concatenate(src_pos, axis=1)
            rec_pos = np.concatenate(rec_pos, axis=1)

        else:
            # Add receiver.
            if self.gmin:
                gmin = self.gmin
            else:
                gmin = self.Npad + 20*self.model.dh
            if self.gmax:
                gmax = self.gmax
            else:
                gmax = gmin + ng * dg

            # Add sources.
            if dg == 3:
                sx = [gmin-20, gmin-5, gmax+5, gmax+20]
            elif dg == 1:
                sx = [gmin-10, gmin-3, gmax+3, gmax+10]
            elif dg == 0.5:
                sx = [gmin-5, gmax+5]
            else:
                raise ValueError("Geophone spacing (dg) must be 3, 1 or 0.5 m")

            # Set source.
            sz = np.full_like(sx, self.source_depth)
            sid = np.arange(0, len(sx))

            src_pos = np.stack([sx,
                                np.zeros_like(sx),
                                sz,
                                sid,
                                np.full_like(sx, self.sourcetype)], axis=0)

            # Set receivers
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
                                np.zeros_like(gx)], axis=0)

        return src_pos, rec_pos


class AcquisitionPermafrost(Acquisition):
    """
    Fix the survey geometry for the ARAC05 survey on the Beaufort Sea.
    """

    def set_rec_src(self):
        # Source and receiver positions.
        offmin = 182        # in meters
        offmax = offmin + 120*self.dg*self.model.dh   # in meters
        if self.singleshot:
            # Add just one source at the right (offmax)
            sx = np.arange(self.Npad + offmax, 1 + self.Npad + offmax)
            # sx = np.arange(self.model.NX / 2,
            #                1 + self.model.NX / 2) * self.model.dh
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

        gx0 = np.arange(offmin, offmax, self.dg*self.model.dh)
        gx = np.concatenate([s - gx0 for s in sx], axis = 0)

        # gx0 = np.arange(gmin, gmax, self.dg) * self.model.dh
        # gx = np.concatenate([gx0 for _ in sx], axis=0)
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


class SeismicGenerator(SeisCL):
    """
    Generate seismic data with SeisCL.
    """

    def __init__(self, acquire: Acquisition, model: BaseModelGenerator,
                 workdir="workdir", gpu=0):
        """
        :param acquire: Parameters for data creation.
        :type acquire: Acquisition
        :param model: Model generator.
        :param model: VelocityModelGenerator
        :param workdir: Working directory for SeisCL. Must be unique for each
                        SeismicGenerator object working in parallel.
        :param gpu: The GPU ID on which to compute data.
        """
        super().__init__()

        self.acquire = acquire
        self.model = model
        # Remove the old working directory and assign a new one.
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
        self.N = np.array([model.NZ, model.NX])
        self.ND = 2
        self.dh = model.dh  # Grid spacing
        self.nab = acquire.Npad  # Set padding cells
        self.dt = acquire.dt  # Time step size
        self.NT = acquire.NT  # Nb of time steps
        self.f0 = acquire.peak_freq  # Source frequency
        self.seisout = acquire.rectype  # Output pressure
        self.freesurf = int(acquire.fs)  # Free surface

        # Assign the GPU to SeisCL.
        nouse = np.arange(0, 16)
        nouse = nouse[nouse != gpu]
        self.no_use_GPUs = nouse

        self.src_pos_all, self.rec_pos_all = acquire.set_rec_src()
        self.resampling = acquire.resampling

        self.wavelet_generator = acquire.source_generator()

    def compute_data(self, props: dict):
        """
        Compute the seismic data of a model.

        :param props: A dictionary of properties' name-values pairs.

        :return: An array containing the modeled seismic data.
        """
        self.src_all = None  # Reset source to generate new random source.
        self.set_forward(self.src_pos_all[3, :], props, withgrad=False)
        self.execute()
        data = self.read_data()
        data = data[0][::self.resampling, :]  # Resample data to reduce space.

        return data
