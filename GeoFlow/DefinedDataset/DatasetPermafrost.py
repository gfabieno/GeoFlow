# -*- coding: utf-8 -*-

import numpy as np

from GeoFlow import GeoDataset
from GeoFlow import Acquisition
from GeoFlow import Vpdepth, Vsdepth, ShotGather, Dispersion
from GeoFlow import EarthModel
from ModelGenerator import (Property, Lithology, Deformation, Sequence,
                            Stratigraphy)


class PermafrostModel(EarthModel):
    def build_stratigraphy(self):
        lithologies = {}

        name = "Water"
        vp = Property("vp", vmin=1430, vmax=1430)
        vpvs = Property("vpvs", vmin=0, vmax=0)
        rho = Property("rho", vmin=1000, vmax=1000)
        q = Property("q", vmin=1000, vmax=1000)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen sediments"  # Buckingham 1996, Fig 11.
        vp = Property("vp", vmin=1700, vmax=1700, texture=200)
        vpvs = Property("vpvs", vmin=4.25, vmax=4.25, texture=1.52)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=50, vmax=50, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Sands"  # Matsushima 2016, fig 13c @ 6C Dou 2016.
        vp = Property("vp", vmin=3700, vmax=3700, texture=200)
        vpvs = Property("vpvs", vmin=2.31, vmax=2.31, texture=0.42)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=60, vmax=60, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Partially Frozen Sands"  # Matsushima 2016, fig 13c @ 3C.
        vp = Property("vp", vmin=3700, vmax=3700, texture=200)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.28)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=10, vmax=10, texture=3.5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Silts"  # Dou 2016, Fig 9, # Buckingham 1996, Fig 11.
        vp = Property("vp", vmin=3400, vmax=3400, texture=300)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.29)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=45, vmax=45, texture=31.5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9; Buckingham 1996, Fig 11.
        name = "Partially Frozen Silts"
        vp = Property("vp", vmin=2200, vmax=2200, texture=450)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=20, vmax=20, texture=10)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9; Buckingham 1996, Fig 11.
        name = "Partially Frozen Silts2"
        vp = Property("vp", vmin=1950, vmax=1950, texture=550)
        vpvs = Property("vpvs", vmin=3, vmax=3, texture=1.3)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Shale"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43.
        vp = Property("vp", vmin=3000, vmax=3000, texture=950)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.87)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=100, vmax=100, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Iperk"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43.
        vp = Property("vp", vmin=4000, vmax=4000, texture=1500)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.7)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=100, vmax=100, texture=30)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen Shale"  # Bellefleur 2007, Figure 3 zone 2.
        # IOE Taglu D-43
        vp = Property("vp", vmin=2200, vmax=2200, texture=200)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.3)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # King, 1976.
        q = Property("q", vmin=70, vmax=100, texture=20)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Matsushima 2016, fig 13c @ 6C Dou 2016.
        name = "Frozen Sands2"
        vp = Property("vp", vmin=2600, vmax=2600, texture=300)
        vpvs = Property("vpvs", vmin=2.6, vmax=2.6, texture=0.6)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=10)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Partially frozen Silts Dou 2016, Fig 9.
        # Buckingham 1996, Fig 11.
        name = "Hydrates"
        vp = Property("vp", vmin=2200, vmax=2200, texture=450)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=20, vmax=20, texture=5)
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        deform = Deformation(max_deform_freq=0.02,
                             min_deform_freq=0.0001,
                             amp_max=6,
                             max_deform_nfreq=40,
                             prob_deform_change=0.4)

        water = Sequence(lithologies=[lithologies["Water"]],
                         thick_min=20, thick_max=40)
        unfrozen1 = Sequence(lithologies=[lithologies["Unfrozen sediments"]],
                             deform=deform, thick_min=8, thick_max=20)
        lithologies_permafrost = [lithologies["Partially Frozen Silts"],
                                  lithologies["Frozen Sands2"],
                                  lithologies["Partially Frozen Silts"]]
        permafrost = Sequence(lithologies=lithologies_permafrost,
                              ordered=False, deform=deform,
                              thick_min=80, thick_max=240)
        unfrozen2 = Sequence(lithologies=[lithologies["Unfrozen sediments"]],
                             deform=deform, thick_min=8, thick_max=40)
        hydrates = Sequence(lithologies=[lithologies["Hydrates"]],
                            deform=deform, thick_min=8, thick_max=80)
        unfrozen3 = Sequence(lithologies=[lithologies["Unfrozen sediments"]],
                             deform=deform, thick_min=8)
        sequences = [water, unfrozen1, permafrost, unfrozen2, hydrates,
                     unfrozen3]
        strati = Stratigraphy(sequences=sequences)

        properties = strati.properties()
        vmin = 99999
        vmax = 0
        for seq in sequences:
            for lith in seq:
                if lith.vpvs.max == 0:
                    vmin = 0
                elif vmin > lith.vp.min / lith.vpvs.max:
                    vmin = lith.vp.min / lith.vpvs.max
                if lith.vpvs.min != 0 and vmax < lith.vp.max/lith.vpvs.min:
                    vmax = lith.vp.max / lith.vpvs.min
        properties["vs"] = [vmin, vmax]

        return strati, properties

    def generate_model(self, seed=None):
        props2d, layerids, layers = super().generate_model(seed=seed)
        tempvpvs = props2d["vpvs"]
        tempvp = props2d["vp"]
        tempvs = tempvp*0
        mask = tempvpvs != 0
        tempvs[mask] = tempvp[mask] / tempvpvs[mask]
        props2d["vs"] = tempvs

        return props2d, layerids, layers


class AcquisitionPermafrost(Acquisition):
    """
    Fix the survey geometry for the ARAC05 survey on the Beaufort Sea.
    """

    def set_rec_src(self):
        # Source and receiver positions.
        offmin = 182  # In meters.
        offmax = offmin + 120*self.dg*self.model.dh  # In meters.
        if self.singleshot:
            # Add just one source at the right (offmax).
            sx = np.arange(self.Npad + offmax, 1 + self.Npad + offmax)
        else:
            # Compute several sources.
            l1 = self.Npad + 1
            l2 = self.model.NX - self.Npad
            sx = np.arange(l1, l2, self.ds) * self.model.dh
        sz = np.full_like(sx, self.source_depth)
        sid = np.arange(0, sx.shape[0])

        src_pos = np.stack([sx,
                            np.zeros_like(sx),
                            sz,
                            sid,
                            np.full_like(sx, self.sourcetype)],
                           axis=0)

        gx0 = np.arange(offmin, offmax, self.dg*self.model.dh)
        gx = np.concatenate([s - gx0 for s in sx], axis=0)

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
                            np.zeros_like(gx)],
                           axis=0)

        return src_pos, rec_pos


class DatasetPermafrost(GeoDataset):
    def set_dataset(self):
        self.trainsize = 5
        self.validatesize = 0
        self.testsize = 0

        model = PermafrostModel()

        model.dh = dh = 2.5
        nshots = 1
        dshots = 50
        length = nshots*dshots + 1682
        z = 1000
        model.NX = int(length / dh)
        model.NZ = int(z / dh)
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ / 2

        model.dip_0 = True
        model.dip_max = 0
        model.ddip_max = 0

        model.layer_num_min = 10
        model.layer_dh_min = 5

        acquire = AcquisitionPermafrost(model=model)
        acquire.peak_freq = 40
        acquire.dt = dt = 2e-4
        acquire.NT = int(2 / dt)
        acquire.dg = 5  # 5 * dh = 12.5 m.
        acquire.fs = True
        acquire.source_depth = 12.5
        acquire.receiver_depth = 12.5

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire),
                  Dispersion.name: Dispersion(model=model, acquire=acquire,
                                              cmax=5000, cmin=1000)}
        outputs = {Vsdepth.name: Vsdepth(model=model, acquire=acquire),
                   Vpdepth.name: Vpdepth(model=model, acquire=acquire)}

        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True
            outputs[name].identify_direct = False

        return model, acquire, inputs, outputs


class DatasetPermafrostNoise(DatasetPermafrost):
    name = "DatasetPermafrost"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        for input in inputs.values():
            input.random_static = True
            input.random_static_max = 1
            input.random_noise = True
            input.random_noise_max = 0.02
        return model, acquire, inputs, outputs


if __name__ == "__main__":
    dataset = DatasetPermafrost()
    dataset.model.animated_dataset()
