# -*- coding: utf-8 -*-

import numpy as np

from GeoFlow import GeoDataset, EarthModel, Acquisition
from GeoFlow.GraphIO import Vsdepth, ShotGather, Dispersion
from ModelGenerator import (Sequence, Stratigraphy, Deformation,
                            Property, Lithology)


class MaswModel(EarthModel):
    def build_stratigraphy(self):
        dh = self.dh

        name = "unsaturated_sand"
        vp = Property("vp", vmin=300, vmax=500, texture=100)
        vpvs = Property("vpvs", vmin=1.8, vmax=2.5, texture=0.2)
        rho = Property("rho", vmin=1500, vmax=1800, texture=50)
        q = Property("q", vmin=7, vmax=20, texture=4)
        unsaturated_sand = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "saturated_sand"
        vp = Property("vp", vmin=1400, vmax=1800, texture=50)
        vpvs = Property("vpvs", vmin=3.5, vmax=12, texture=1)
        rho = Property("rho", vmin=1800, vmax=2200, texture=50)
        q = Property("q", vmin=7, vmax=20, texture=4)
        saturated_sand = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "saturated_clay"
        vp = Property("vp", vmin=1500, vmax=1800, texture=50)
        vpvs = Property("vpvs", vmin=6, vmax=20, texture=1)
        rho = Property("rho", vmin=1800, vmax=2200, texture=50)
        q = Property("q", vmin=15, vmax=30, texture=4)
        saturated_clay = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "weathered_shale"
        vp = Property("vp", vmin=1950, vmax=2100, texture=50)
        vpvs = Property("vpvs", vmin=2.4, vmax=4.5, texture=1)
        rho = Property("rho", vmin=2000, vmax=2400, texture=50)
        q = Property("q", vmin=15, vmax=30, texture=4)
        weathered_shale = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "shale"
        vp = Property("vp", vmin=2000, vmax=2500, texture=20)
        vpvs = Property("vpvs", vmin=2.6, vmax=4.5, texture=1)
        rho = Property("rho", vmin=2000, vmax=2400, texture=50)
        q = Property("q", vmin=30, vmax=60, texture=4)
        shale = Lithology(name=name, properties=[vp, vpvs, rho, q])

        deform = Deformation(max_deform_freq=0.02,
                             min_deform_freq=0.0001,
                             amp_max=5,
                             max_deform_nfreq=10,
                             prob_deform_change=0.1)
        deform = None

        unsat_seq = Sequence(name="unsaturated",
                             lithologies=[unsaturated_sand],
                             thick_max=int(25/dh), deform=deform)
        sat_seq = Sequence(name="saturated",
                           lithologies=[saturated_clay,
                                        saturated_sand],
                           thick_max=int(100/dh), deform=deform)
        weathered_seq = Sequence(name="weathered",
                                 lithologies=[weathered_shale],
                                 thick_max=int(50/dh), deform=deform)
        roc_seq = Sequence(name="roc",
                           lithologies=[shale],
                           thick_max=99999, deform=deform)

        sequences = [unsat_seq,
                     sat_seq,
                     weathered_seq,
                     roc_seq]
        strati = Stratigraphy(sequences=sequences)

        properties = strati.properties()
        vmin = 99999
        vmax = 0
        for seq in sequences:
            for lith in seq:
                if vmin > lith.vp.min / lith.vpvs.max:
                    vmin = lith.vp.min / lith.vpvs.max
                if vmax < lith.vp.max / lith.vpvs.min:
                    vmax = lith.vp.max / lith.vpvs.min
        properties["vs"] = [vmin, vmax]

        return strati, properties

    def generate_model(self, seed=None):
        props2D, layerids, layers = super().generate_model(seed=seed)
        props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers


class AquisitionMASW(Acquisition):
    def set_rec_src(self):
        # In meters.
        dh = self.model.dh
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
                gmin = (self.Npad+10)*dh + 20
            if self.gmax:
                gmax = self.gmax
            else:
                gmax = gmin + ng * dg

            # Add sources.
            if self.singleshot:
                if dg == 3:
                    sx = [gmin - 20]
                elif dg == 1:
                    sx = [gmin - 10]
                elif dg == 0.5:
                    sx = [gmin - 5]
            else:
                if dg == 3:
                    sx = [gmin-20, gmin-5, gmax+5, gmax+20]
                elif dg == 1:
                    sx = [gmin-10, gmin-3, gmax+3, gmax+10]
                elif dg == 0.5:
                    sx = [gmin-5, gmax+5]
                else:
                    raise ValueError("Geophone spacing (dg) must be 3, 1 or "
                                     "0.5 m")

            # Set source.
            sz = np.full_like(sx, self.source_depth)
            sid = np.arange(0, len(sx))

            src_pos = np.stack([sx,
                                np.zeros_like(sx),
                                sz,
                                sid,
                                np.full_like(sx, self.sourcetype)], axis=0)

            # Set receivers.
            gx0 = np.arange(gmin, gmax, dg)
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


class DatasetMASW(GeoDataset):
    name = "Dataset_masw"

    def set_dataset(self):
        self.trainsize = 700
        self.validatesize = 150
        self.testsize = 150

        nab = 250

        model = MaswModel()
        model.dh = dh = 0.1
        model.NX = int(120/dh + 2*nab)
        model.NZ = int(50/dh + nab)

        model.marine = False
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ / 2

        model.dip_0 = True
        model.dip_max = 0
        model.ddip_max = 0

        model.layer_num_min = 1
        model.layer_dh_min = int(1/dh)
        model.layer_dh_max = int(8/dh)

        acquire = AquisitionMASW(model=model)
        acquire.peak_freq = 35
        acquire.sourcetype = 2  # Force in z (2).
        acquire.ds = 5
        acquire.dt = dt = 0.00002
        acquire.NT = int(1.5 / dt)  # 2 s survey.
        acquire.dg = 1  # 3m spacing.
        acquire.fs = True  # Free surface.
        acquire.source_depth = 0
        acquire.receiver_depth = 0
        acquire.rectype = 1
        acquire.singleshot = True
        acquire.Npad = nab
        acquire.abs_type = 2

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire),
                  Dispersion.name: Dispersion(model=model, acquire=acquire,
                                              cmax=1500, cmin=100, fmin=0,
                                              fmax=50)}
        outputs = {Vsdepth.name: Vsdepth(model=model, acquire=acquire)}
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True

        return model, acquire, inputs, outputs


class DatasetMASWNoise(DatasetMASW):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        for input in inputs.values():
            input.random_static = True
            input.random_static_max = 1
            input.random_noise = True
            input.random_noise_max = 0.02
        return model, acquire, inputs, outputs


if __name__ == "__main__":
    np.random.seed(0)
    dataset = DatasetMASW()
    dataset.trainsize = 5
    dataset.validatesize = 0
    dataset.testsize = 0
    dataset.generate_dataset(gpus=1)
    dataset.animate()
