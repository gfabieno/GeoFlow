#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define parameters for different cases"""

from vrmslearn.Case import Case
from vrmslearn.VelocityModelGenerator import (MarineModelGenerator,
                                              MaswModelGenerator,
                                              PermafrostModelGenerator)
from vrmslearn.SeismicGenerator import Acquisition, AcquisitionPermafrost
from vrmslearn.LabelGenerator import LabelGenerator, PermafrostLabelGenerator
import argparse

class Case_Permafrost(Case):

    name = "Case_Permafrost"

    def __init__(self, noise=0):
        if noise == 1:
            self.name = self.name + "_noise"
        super().__init__()
        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02

    def set_case(self):
        self.trainsize = 5
        self.validatesize = 0
        self.testsize = 0

        model = PermafrostModelGenerator()

        model.dh = dh = 2.5
        Nshots = 1
        dshots = 50
        length = Nshots*dshots + 1682
        z = 1000
        model.NX = int(length/dh)
        model.NZ = int(z/dh)

        model.marine = False    #??
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ/2

        model.dip_0 = True
        model.dip_max = 0
        model.ddip_max = 0

        model.layer_num_min = 3
        model.layer_dh_min = 20
        # model.layer_dh_max = 20

        model.Dispersion = True

        acquire = AcquisitionPermafrost(model=model)
        acquire.peak_freq = 40
        # acquire.sourcetype = 2
        acquire.dt = dt = 2e-4
        acquire.NT = int(2/dt)
        acquire.dg = dg = 5             # 5*dh = 12.5 m
        # acquire.gmin = int(100 / dh)
        # acquire.gmax = int(acquire.gmin*dg)
        acquire.fs = True
        acquire.source_depth = 12.5
        acquire.receiver_depth = 12.5
        # acquire.rectype = 1

        # label = LabelGenerator(model=model, acquire=acquire)
        label = PermafrostLabelGenerator(model=model, acquire=acquire)
        label.identify_direct = False
        label.train_on_shots = True
        label.label_names = ('vp','vs')
        label.weight_names = ['tweight', 'dweight']

        return model, acquire, label


class Case_masw(Case):

    name = "Case_masw"

    def __init__(self, noise=0):

        if noise == 1:
            self.name = self.name + "_noise"
        super().__init__()
        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02

    def set_case(self):

        model = MaswModelGenerator()
        model.NX = 500
        model.NZ = 100
        model.dh = dh = 1

        model.marine = False
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ / 2

        model.dip_0 = True
        model.dip_max = 0
        model.ddip_max = 0

        model.layer_num_min = 1
        model.layer_dh_min = 5
        model.layer_dh_max = 20

        acquire = Acquisition(model=model)
        acquire.peak_freq = 26
        acquire.sourcetype = 2
        acquire.dt = dt = 0.0001
        acquire.NT = int(2 / dt)
        acquire.dg = dg = 3
        acquire.gmin = int(100 / dh)
        acquire.gmax = int(acquire.gmin*dg)
        acquire.fs = True
        acquire.source_depth = 0
        acquire.receiver_depth = 0
        acquire.rectype = 1

        label = LabelGenerator(model=model, acquire=acquire)
        label.identify_direct = False
        label.train_on_shots = True

        return model, acquire, label


class Case1Dsmall(Case):

    name = "Case1Dsmall"

    def set_case(self):

        model, acquire, label = super().set_case()
        label.train_on_shots = True

        return model, acquire, label


class Case1Darticle(Case):

    name = "Case1Darticle"

    def set_case(self):

        model = MarineModelGenerator()
        model.layer_dh_min = 5
        model.layer_num_min = 48
        model.dh = dh = 6.25
        model.NX = 692 * 2
        model.NZ = 752 * 2
        model.water_vmin = 1430
        model.water_vmax = 1560
        model.water_dmin = 2500
        model.water_dmax = 4500
        model.vp_min = 1300.0
        model.vp_max = 4000.0

        acquire = Acquisition(model=model)
        acquire.peak_freq = 26
        acquire.df = 5
        acquire.wavefuns = [0, 1]
        acquire.dt = dt = 0.0004
        acquire.NT = int(8.0 / dt)
        acquire.resampling = 10
        acquire.dg = dg = 8
        acquire.gmin = int(470 / dh)
        acquire.gmax = int((470 + 72 * dg * dh) / dh)
        acquire.minoffset = 470
        acquire.fs = False
        acquire.source_depth = (acquire.Npad + 4) * dh
        acquire.receiver_depth = (acquire.Npad + 4) * dh

        label = LabelGenerator(model=model, acquire=acquire)
        label.identify_direct = False
        label.train_on_shots = True

        return model, acquire, label

    def __init__(self, noise=0):

        if noise == 1:
            self.name = self.name + "_noise"
        super().__init__()
        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02


class Case2Dtest(Case):

    name = "Case2Dtest"

    def set_case(self):

        model = MarineModelGenerator()
        model.NX = 150
        model.NZ = 100
        model.water_dmin = 300
        model.water_dmax = 600
        model.vp_trend_min = 0
        model.vp_trend_max = 2
        model.max_deform_freq = 0.06
        model.min_deform_freq = 0.0001
        model.amp_max = 8
        model.max_deform_nfreq = 40
        model.prob_deform_change = 0.7
        model.dip_max = 20
        model.ddip_max = 10
        model.layer_num_min = 5
        model.layer_dh_min = 10

        acquire = Acquisition(model=model)
        acquire.NT = 2560
        acquire.dg = 5
        acquire.ds = 5
        acquire.gmin = acquire.dg
        acquire.gmax = model.NX - acquire.gmin
        acquire.singleshot = False

        label = LabelGenerator(model=model, acquire=acquire)
        label.train_on_shots = True

        return model, acquire, label

    def __init__(self, noise=0):

        if noise == 1:
            self.name = self.name + "_noise"

        super().__init__()
        if noise == 1:
            self.label.random_static = True
            self.label.random_static_max = 1
            self.label.random_noise = True
            self.label.random_noise_max = 0.02


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        # default="Case1Dsmall",
        default="Case_Permafrost",
        help="Name of the case to use"
    )
    args, unparsed = parser.parse_known_args()

    case = eval(args.case)()
    case.model.animated_dataset()
