#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Defines parameters for different cases
"""

from vrmslearn.ModelParameters import ModelParameters
from vrmslearn.ModelGenerator import ModelGenerator
import matplotlib.pyplot as plt

def Case_1Dsmall():
    return ModelParameters()

def Case_1Darticle(noise=0):
    pars = ModelParameters()
    pars.layer_dh_min = 5
    pars.layer_num_min = 48
    
    pars.dh = 6.25
    pars.peak_freq = 26
    pars.df = 5
    pars.wavefuns = [0, 1]
    pars.NX = 692*2
    pars.NZ = 752*2
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
    
    pars.mute_dir = True
    if noise == 1:
        pars.random_static = True
        pars.random_static_max = 1
        pars.random_noise = True
        pars.random_noise_max = 0.02

    return pars


def Case_2Dsmall(noise=0):

    pars = ModelParameters()
    pars.layer_dh_min = 20
    pars.num_layers = 0
    pars.marine = True
    pars.water_dmin = 100
    pars.water_dmax = 1000
    pars.vp_trend_min = 0
    pars.vp_trend_max = 2

    pars.max_deform_freq = 0.1  # Max frequency of the layer boundary function
    pars.min_deform_freq = 0.0001  # Min frequency of the layer boundary function
    pars.amp_max = 26  # Maximum amplitude of boundary deformations
    pars.max_deform_nfreq = 40  # Maximum nb of frequencies of boundary
    pars.prob_deform_change = 0.7  # Probability that a boundary shape will change
    pars.angle_max = 20

    pars.num_layers = 0
    pars.layer_num_min = 15
    pars.layer_dh_min = 10
    pars.NT = 10000

    pars.marine = True
    pars.flat = False
    if noise == 1:
        pars.random_static = True
        pars.random_static_max = 1
        pars.random_noise = True
        pars.random_noise_max = 0.02

    return pars


def plot_models(pars):
    gen = ModelGenerator(pars)
    vp, vs, rho = gen.generate_model()
    plt.imshow(vp)
    plt.show()


if __name__ == "__main__":

    pars = Case_2Dsmall()
    plot_models(pars)
