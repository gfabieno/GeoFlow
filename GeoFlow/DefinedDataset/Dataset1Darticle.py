# -*- coding: utf-8 -*-

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import (Reftime, Vrms, Vint, Vdepth, ShotGather)


class Dataset1Darticle(GeoDataset):
    def __init__(self, noise=False):
        super().__init__()
        if noise:
            for input in self.inputs.values():
                input.random_static = True
                input.random_static_max = 1
                input.random_noise = True
                input.random_noise_max = 0.02

    def set_dataset(self):
        model = MarineModel()
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

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {Reftime.name: Reftime(model=model, acquire=acquire),
                   Vrms.name: Vrms(model=model, acquire=acquire),
                   Vint.name: Vint(model=model, acquire=acquire),
                   Vdepth.name: Vdepth(model=model, acquire=acquire)}
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True
            outputs[name].identify_direct = False
        return model, acquire, inputs, outputs


if __name__ == "__main__":
    dataset = Dataset1Darticle()
    dataset.model.animated_dataset()
