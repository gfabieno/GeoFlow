#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define parameters for different cases"""

from vrmslearn.Case import Case
from vrmslearn.VelocityModelGenerator import MarineModelGenerator
import argparse


class Case1Dsmall(Case):

    name = "Case1Dsmall"

    def set_case(self):
        super().set_case()
        self.label.train_on_shots = True


class Case1Darticle(Case):

    name = "Case1Darticle"
    Model = MarineModelGenerator

    def set_case(self):

        self.model.layer_dh_min = 5
        self.model.layer_num_min = 48
        self.model.dh = dh = 6.25
        self.model.NX = 692 * 2
        self.model.NZ = 752 * 2
        self.model.water_vmin = 1430
        self.model.water_vmax = 1560
        self.model.water_dmin = 2500
        self.model.water_dmax = 4500
        self.model.vp_min = 1300.0
        self.model.vp_max = 4000.0

        self.acquire.peak_freq = 26
        self.acquire.df = 5
        self.acquire.wavefuns = [0, 1]
        self.acquire.dt = dt = 0.0004
        self.acquire.NT = int(8.0 / dt)
        self.acquire.resampling = 10
        self.acquire.dg = dg = 8
        self.acquire.gmin = int(470 / dh)
        self.acquire.gmax = int((470 + 72 * dg * dh) / dh)
        self.acquire.minoffset = 470
        self.acquire.fs = False
        self.acquire.source_depth = (self.acquire.Npad + 4) * dh
        self.acquire.receiver_depth = (self.acquire.Npad + 4) * dh

        self.label.identify_direct = False
        self.label.train_on_shots = True

    def __init__(self, trainsize=1, validatesize=0, testsize=0, noise=0):

        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02
            self.name = self.name + "_noise"
        super().__init__(
            trainsize=trainsize,
            validatesize=validatesize,
            testsize=testsize,
        )


class Case2Dtest(Case):

    name = "Case2Dtest"
    Model = MarineModelGenerator

    def set_case(self):

        self.model.NX = 150
        self.model.NZ = 100
        self.model.water_dmin = 300
        self.model.water_dmax = 600
        self.model.vp_trend_min = 0
        self.model.vp_trend_max = 2

        self.model.max_deform_freq = 0.06
        self.model.min_deform_freq = 0.0001
        self.model.amp_max = 8
        self.model.max_deform_nfreq = 40
        self.model.prob_deform_change = 0.7
        self.model.dip_max = 20
        self.model.ddip_max = 10

        self.model.layer_num_min = 5
        self.model.layer_dh_min = 10

        self.acquire.NT = 2560

        self.acquire.dg = 5
        self.acquire.ds = 5
        self.acquire.gmin = self.acquire.dg
        self.acquire.gmax = self.model.NX - self.acquire.gmin

        self.acquire.singleshot = False
        self.label.train_on_shots = True

    def __init__(self, trainsize=1005, validatesize=0, testsize=0, noise=0):

        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02
            self.name = self.name + "_noise"
        super().__init__(
            trainsize=trainsize,
            validatesize=validatesize,
            testsize=testsize,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default="Case_1Dsmall",
        help="Name of the case to use"
    )
    args, unparsed = parser.parse_known_args()

    case = eval(args.case)()
    case.model.animated_dataset()
