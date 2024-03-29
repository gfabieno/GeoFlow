# -*- coding: utf-8 -*-
"""
Generate the labels (seismic data) using SeisCL.

Requires the SeisCL python interface.
"""

import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from SeisCL.SeisCL import SeisCL
from GeoFlow.SeismicUtilities import random_wavelet_generator
from GeoFlow.EarthModel import EarthModel


class Acquisition:
    """
    Define all model parameters needed to model seismic data.
    """

    def __init__(self, model: EarthModel):
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
        # Absorbing boundary type:
        # 1) CPML or 2) absorbing layer of Cerjan.
        self.abs_type = 1

        self.singleshot = True
        # Whether to fill the surface with geophones or to use inline spread.
        # Either `'full'` or `'inline'`.
        self.configuration = 'full'
        # Number of attenuation mechanism (L=0 elastic)
        self.L = 0
        # Frequencies of the attenuation mechanism
        self.FL = np.array(15)

    def set_rec_src(self):
        """
        Provide the sources' and receivers' positions for SeisCL.

        Override to change which data is modelled if needed.

        :return:
            src_pos: Source array.
            rec_pos: Receiver array.
        """
        assert self.configuration in ['inline', 'full']

        if self.singleshot:
            # Add just one source in the middle
            middle = self.model.NX / 2 * self.model.dh
            sx = np.array([middle])
        elif self.configuration == 'inline':
            # Compute several sources
            start_idx = self.Npad + 1
            if self.gmin is not None and self.gmin < 0:
                start_idx += -self.gmin
            end_idx = self.model.NX - self.Npad
            if self.gmax is not None and self.gmax > 0:
                end_idx += -self.gmax
            sx = np.arange(start_idx, end_idx, self.ds) * self.model.dh
        elif self.configuration == 'full':
            # Compute several sources
            start_idx = self.Npad + 1
            end_idx = self.model.NX - self.Npad
            sx = np.arange(start_idx, end_idx, self.ds) * self.model.dh
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, sx.shape[0])

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)], axis=0)

        # Add receivers
        if self.gmin is not None:
            gmin = self.gmin
        else:
            if self.configuration == 'inline':
                gmin = -(self.model.NX-2*self.Npad) // 2
            elif self.configuration == 'full':
                gmin = self.Npad
        if self.gmax is not None:
            gmax = self.gmax
        else:
            if self.configuration == 'inline':
                gmax = (self.model.NX-2*self.Npad) // 2
            elif self.configuration == 'full':
                gmax = self.model.NX - self.Npad

        gx0 = np.arange(gmin, gmax, self.dg) * self.model.dh
        gsid = np.concatenate([np.full_like(gx0, s) for s in sid], axis=0)
        if self.configuration == 'inline':
            gx = np.concatenate([s + gx0 for s in sx], axis=0)
        elif self.configuration == 'full':
            gx = np.concatenate([gx0 for _ in sx], axis=0)
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

    def plot_acquisition_geometry(self):
        src_pos, rec_pos = self.set_rec_src()
        qty_srcs = src_pos.shape[1]
        src_x, _, _, src_id, _ = src_pos
        rec_x, _, _, rec_src_id, _, _, _, _ = rec_pos
        props, _, _ = self.model.generate_model()
        model = props['vp']
        x_max = self.model.NX * self.model.dh
        z_max = self.model.NZ * self.model.dh

        height_geometry = 12 * (qty_srcs+1) / 72
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                figsize=(8, 8+height_geometry),
                                gridspec_kw={'height_ratios': [height_geometry,
                                                               8],
                                             'hspace': 0})

        axs[0].scatter(src_x, src_id, marker='x', s=8, label="Sources")
        axs[0].scatter(rec_x, rec_src_id, marker='v', s=8, label="Receivers")
        axs[0].set_axis_off()
        axs[0].set_xlim([0, x_max])
        axs[0].set_ylim([-.5, qty_srcs-.5])
        axs[0].legend()

        axs[1].imshow(model, origin='upper', extent=[0, x_max, z_max, 0],
                      aspect='auto')
        axs[1].set_ylabel("Depth (m)")
        axs[1].set_xlabel("Position (m)")

        plt.tight_layout()
        plt.show()


class SeismicGenerator(SeisCL):
    """
    Generate seismic data with SeisCL.
    """

    def __init__(self, acquire: Acquisition, model: EarthModel,
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
        self.abs_type = acquire.abs_type # Absorbing boundary type
        self.L = acquire.L  # Number of attenuation mechanism (L=0 elastic)
        self.FL = acquire.FL  # Frequencies of the attenuation mechanism

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
