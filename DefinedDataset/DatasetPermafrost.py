# -*- coding: utf-8 -*-

import numpy as np
from GeoFlow import GeoDataset
from GeoFlow import Acquisition
from GeoFlow import Vsdepth, ShotGather, Dispersion
from GeoFlow import EarthModel
from ModelGenerator import (Property,
                            Lithology,
                            Deformation,
                            Sequence,
                            Stratigraphy)


class PermafrostModel(EarthModel):
    def __init__(self):
        super().__init__()

    def build_stratigraphy(self):
        lithologies = {}

        name = "Water"
        vp = Property("vp", vmin=1430, vmax=1430)
        # vs = Property("vs", vmin=0, vmax=0)
        vpvs = Property("vpvs", vmin=0, vmax=0)
        rho = Property("rho", vmin=1000, vmax=1000)
        q = Property("q", vmin=1000, vmax=1000)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen sediments"  # Buckingham 1996, Fig 11
        vp = Property("vp", vmin=1700, vmax=1700, texture=200)
        # vs = Property("vs", vmin=400, vmax=400, texture=150)
        vpvs = Property("vpvs", vmin=4.25, vmax=4.25, texture=1.52)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=50, vmax=50, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Sands"  # Matsushima 2016, fig 13c @ 6C Dou 2016
        vp = Property("vp", vmin=3700, vmax=3700, texture=200)
        # vs = Property("vs", vmin=1600, vmax=1600, texture=250)
        vpvs = Property("vpvs", vmin=2.31, vmax=2.31, texture=0.42)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=60, vmax=60, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Partially Frozen Sands"  # Matsushima 2016, fig 13c @ 3C
        vp = Property("vp", vmin=3700, vmax=3700, texture=200)
        # vs = Property("vs", vmin=1332, vmax=1332, texture=70)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.28)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=10, vmax=10, texture=3.5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Silts"  # Dou 2016, Fig 9, # Buckingham 1996, Fig 11
        vp = Property("vp", vmin=3400, vmax=3400, texture=300)
        # vs = Property("vs", vmin=1888, vmax=1888, texture=170)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.29)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=45, vmax=45, texture=31.5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9
        # Buckingham 1996, Fig 11
        name = "Partially Frozen Silts"
        vp = Property("vp", vmin=2200, vmax=2200, texture=450)
        # vs = Property("vs", vmin=792, vmax=792, texture=160)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=20, vmax=20, texture=10)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Dou 2016, Fig 9, #Buckingham 1996, Fig 11
        name = "Partially Frozen Silts2"
        vp = Property("vp", vmin=1950, vmax=1950, texture=550)
        # vs = Property("vs", vmin=650, vmax=650, texture=186)
        vpvs = Property("vpvs", vmin=3, vmax=3, texture=1.3)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Shale"  # Bellefleur 2007, Figure 3 zone 2
        # IOE Taglu D-43
        vp = Property("vp", vmin=3000, vmax=3000, texture=950)
        # vs = Property("vs", vmin=1666, vmax=650, texture=527)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.87)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=100, vmax=100, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Iperk"  # Bellefleur 2007, Figure 3 zone 2
        # IOE Taglu D-43
        vp = Property("vp", vmin=4000, vmax=4000, texture=1500)
        # vs = Property("vs", vmin=2222, vmax=2222, texture=52)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.7)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=100, vmax=100, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen Shale"  # Bellefleur 2007, Figure 3 zone 2
        # IOE Taglu D-43
        vp = Property("vp", vmin=2200, vmax=2200, texture=200)
        # vs = Property("vs", vmin=1222, vmax=1222, texture=111)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.3)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=70, vmax=100, texture=20)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Matsushima 2016, fig 13c @ 6C Dou 2016
        name = "Frozen Sands2"
        vp = Property("vp", vmin=2600, vmax=2600, texture=300)
        # vs = Property("vs", vmin=1000, vmax=1000, texture=150)
        vpvs = Property("vpvs", vmin=2.6, vmax=2.6, texture=0.6)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=10)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        # Modified from Partially frozen Silts Dou 2016, Fig 9,
        # Buckingham 1996, Fig 11
        name = "Hydrates"
        vp = Property("vp", vmin=2200, vmax=2200, texture=450)
        # vs = Property("vs", vmin=792, vmax=792, texture=160)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=20, vmax=20, texture=5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        deform = Deformation(max_deform_freq=0.02,
                             min_deform_freq=0.0001,
                             amp_max=6,
                             max_deform_nfreq=40,
                             prob_deform_change=0.4)

        water = Sequence(lithologies=[lithologies["Water"]],
                         thick_min=50, thick_max=150)
        unfrozen = Sequence(lithologies=[lithologies["Unfrozen sediments"]],
                            deform=deform)
        permafrost = Sequence(lithologies=[lithologies["Partially Frozen Silts"],
                                           lithologies["Frozen Sands2"],
                                           lithologies["Partially Frozen Silts"],
                                           lithologies["Unfrozen sediments"],
                                           lithologies["Hydrates"]],
                              ordered=False, deform=deform)

        sequences = [water, unfrozen, permafrost, unfrozen]
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

        return strati

    def generate_model(self, seed=None):

        props2D, layerids, layers = super().generate_model(seed=seed)
        tempvpvs = props2D["vpvs"]
        tempvp = props2D["vp"]
        tempvs = tempvp*0
        mask = tempvpvs != 0
        tempvs[mask] = tempvp[mask] / tempvpvs[mask]
        props2D["vs"] = tempvs
        # props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers

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

class DatasetPermafrost(GeoDataset):
    name = "DatasetPermafrost"

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
        self.trainsize = 5
        self.validatesize = 0
        self.testsize = 0

        model = PermafrostModel()

        model.dh = dh = 2.5
        Nshots = 1
        dshots = 50
        length = Nshots*dshots + 1682
        z = 1000
        model.NX = int(length/dh)
        model.NZ = int(z/dh)

        model.marine = False  # ??
        model.texture_xrange = 3
        model.texture_zrange = 1.95 * model.NZ/2

        model.dip_0 = True
        model.dip_max = 0
        model.ddip_max = 0

        model.layer_num_min = 3
        model.layer_dh_min = 20
        # model.layer_dh_max = 20

        # TODO That won't work anymore.
        model.Dispersion = True

        acquire = AcquisitionPermafrost(model=model)
        acquire.peak_freq = 40
        # acquire.sourcetype = 2
        acquire.dt = dt = 2e-4
        acquire.NT = int(2/dt)
        acquire.dg = dg = 5  # 5*dh = 12.5 m
        # acquire.gmin = int(100 / dh)
        # acquire.gmax = int(acquire.gmin*dg)
        acquire.fs = True
        acquire.source_depth = 12.5
        acquire.receiver_depth = 12.5
        # acquire.rectype = 1

        # label = LabelGenerator(model=model, acquire=acquire)
        # TODO write GraphInput and GraphOutput for dispersion.
        # label = PermafrostLabelGenerator(model=model, acquire=acquire)
        # label.identify_direct = False
        # label.train_on_shots = True
        # label.label_names = ('vp','vs')
        # label.weight_names = ['tweight', 'dweight']
        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire),
                  Dispersion.name: Dispersion(model=model, acquire=acquire,
                                              cmax=5000, cmin=1000)}
        outputs = {Vsdepth.name: Vsdepth(model=model, acquire=acquire)}

        return model, acquire, inputs, outputs

if __name__ == "__main__":
    dataset = DatasetPermafrost()
    dataset.model.animated_dataset()