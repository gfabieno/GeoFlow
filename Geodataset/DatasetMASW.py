# -*- coding: utf-8 -*-


from GeoFlow.Dataset import Dataset
from GeoFlow.EarthModel import MaswModel
from GeoFlow.SeismicGenerator import Aquisition_masw
from GeoFlow.GraphIO import Vsdepth, ShotGather


class DatasetMASW(Dataset):
    name = "Dataset_masw"

    def __init__(self, noise=0):
        if noise == 1:
            self.name = self.name + "_noise"
        super().__init__()
        if noise == 1:
            for name in self.inputs:
                self.inputs[name].random_static = True
                self.inputs[name].random_static_max = 1
                self.inputs[name].random_noise = True
                self.inputs[name].random_noise_max = 0.02

    def set_dataset(self):
        model = MaswModel()
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

        acquire = Aquisition_masw(model=model)
        acquire.peak_freq = 26
        acquire.sourcetype = 2  # Force in z (2).
        acquire.ds = 5
        acquire.dt = dt = 0.0001
        acquire.NT = int(2 / dt)  # 2 s survey.
        acquire.tdelay = dt * 5
        acquire.dg = 'all'  # 3 / dh # 3m spacing
        acquire.fs = True  # Free surface
        acquire.source_depth = 0
        acquire.receiver_depth = 0
        acquire.rectype = 1
        acquire.singleshot = False

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {Vsdepth.name: Vsdepth(model=model, acquire=acquire)}
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True

        return model, acquire, inputs, outputs