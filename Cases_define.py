#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Defines parameters for different cases here.
"""

from vrmslearn.Case import Case, CaseCollection
from vrmslearn.ModelParameters import ModelParameters

class Case_1Dsmall(Case):

    name = "1Dsmall"
    pars = ModelParameters()

class Case_1Darticle(Case):

    name = "1Darticle"
    pars = ModelParameters()
    pars.layer_dh_min = 5
    pars.layer_num_min = 48

    pars.dh = 6.25
    pars.peak_freq = 26
    pars.df = 5
    pars.wavefuns = [0, 1]
    pars.NX = 692 * 2
    pars.NZ = 752 * 2
    pars.dt = 0.0004
    pars.NT = int(8.0 / pars.dt)
    pars.resampling = 10

    pars.dg = 8
    pars.gmin = int(470 / pars.dh)
    pars.gmax = int((470 + 72 * pars.dg * pars.dh) / pars.dh)
    pars.minoffset = 470

    pars.vp_min = 1300.0  # maximum value of vp (in m/s)
    pars.vp_max = 4000.0  # minimum value of vp (in m/s)

    pars.marine = True
    pars.water_vmin = 1430
    pars.water_vmax = 1560
    pars.water_dmin = 2500
    pars.water_dmax = 4500

    pars.fs = False
    pars.source_depth = (pars.Npad + 4) * pars.dh
    pars.receiver_depth = (pars.Npad + 4) * pars.dh
    pars.identify_direct = False

    def __init__(self, trainsize=1, validatesize=0, testsize=0, noise=0):

        if noise == 1:
            self.pars.random_static = True
            self.pars.random_static_max = 1
            self.pars.random_noise = True
            self.pars.random_noise_max = 0.02
        self.name = self.name + "_noise"
        super().__init__(trainsize=trainsize,
                         validatesize=validatesize,
                         testsize=testsize)

class Case_2Dtest(Case):

    name = "2Dtest"
    pars = ModelParameters()

    pars.NX = 350
    pars.NZ = 256

    pars.marine = True

    pars.water_dmin = 300
    pars.water_dmax = 600
    pars.vp_trend_min = 0
    pars.vp_trend_max = 2

    pars.max_deform_freq = 0.06
    pars.min_deform_freq = 0.0001
    pars.amp_max = 8
    pars.max_deform_nfreq = 40
    pars.prob_deform_change = 0.7
    pars.angle_max = 20
    pars.dangle_max = 10  # Maximum dip difference between two adjacent layers

    pars.num_layers = 0
    pars.layer_num_min = 5
    pars.layer_dh_min = 10
    pars.NT = 2560

    pars.dg = 5
    pars.ds = 5
    pars.gmin = pars.dg
    pars.gmax = 120

    pars.flat = False

    def __init__(self, trainsize=1005, validatesize=0, testsize=0, noise=0):

        if noise == 1:
            self.pars.random_static = True
            self.pars.random_static_max = 1
            self.pars.random_noise = True
            self.pars.random_noise_max = 0.02
        super().__init__(trainsize=trainsize,
                         validatesize=validatesize,
                         testsize=testsize)



if __name__ == "__main__":

    case = Case_2Dtest()
    case.plot_model()
