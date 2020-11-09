#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Class to generate seismic models
"""

import argparse
from ModelGenerator import (ModelGenerator, Sequence, Stratigraphy, Deformation,
                            Property, Lithology)


class BaseModelGenerator(ModelGenerator):

    def __init__(self):

        super().__init__()
        # Number of grid cells in X direction.
        self.NX = 256
        # Number of grid cells in Z direction.
        self.NZ = 256
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = 10.0

        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_min = 50
        # Minimum number of layers.
        self.layer_num_min = 3
        # Fix the number of layers if not 0.
        self.num_layers = 0

        # Maximum value of vp (in m/s).
        self.vp_min = 1000.0
        # Minimum value of vp (in m/s).
        self.vp_max = 5000.0
        # Minimum trend for velocity variation in z.
        self.vp_trend_min = 0
        # Maximum trend for velocity variation in z.
        self.vp_trend_max = 0

        # Maximum dip of a layer.
        self.dip_max = 0
        # Maximum dip difference between two adjacent layers.
        self.ddip_max = 5
        # Max frequency of the layer boundary function.
        self.max_deform_freq = 0
        # Min frequency of the layer boundary function.
        self.min_deform_freq = 0
        # Maximum amplitude of boundary oscillations.
        self.amp_max = 25
        # Maximum nb of frequencies of boundary.
        self.max_deform_nfreq = 20
        # Probability that a boundary shape will.
        self.prob_deform_change = 0.3
        # Change between two layers.
        # Add random noise two a layer (% or velocity).
        self.max_texture = 0
        # Range of the filter in x for texture creation.
        self.texture_xrange = 0
        # Range of the filter in z for texture creation.
        self.texture_zrange = 0

        self.strati = None
        # Define if data will be presented in dispersion plots
        self.Dispersion = False

    def generate_model(self, seed=None):
        """
        Output the media parameters required for seismic modelling, in this
        case vp, vs and rho.

        @params:

        @returns:
        vp (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vp.
        vs (numpy.ndarray)  : numpy array (self.pars.NZ, self.pars.NX) for vs.
        rho (numpy.ndarray) : numpy array (self.pars.NZ, self.pars.NX) for rho
                              values.
        """
        if self.strati is None:
            self.strati = self.build_stratigraphy()
        props2D, layerids, layers = super().generate_model(self.strati,
                                                                 seed=seed)

        return props2D, layerids, layers

    def build_stratigraphy(self):

        vp = Property(name="vp", vmin=self.vp_min, vmax=self.vp_max,
                      texture=self.max_texture, trend_min=self.vp_trend_min,
                      trend_max=self.vp_trend_max)
        vs = Property(name="vs", vmin=0, vmax=0)
        rho = Property(name="rho", vmin=2000, vmax=2000)
        lith = Lithology(name='simple', properties=[vp, vs, rho])
        deform = Deformation(max_deform_freq=self.max_deform_freq,
                             min_deform_freq=self.min_deform_freq,
                             amp_max=self.amp_max,
                             max_deform_nfreq=self.max_deform_freq,
                             prob_deform_change=self.prob_deform_change)
        sequence = Sequence(lithologies=[lith], ordered=False, deform=deform)
        strati = Stratigraphy(sequences=[sequence])

        return strati


class MarineModelGenerator(BaseModelGenerator):

    def __init__(self):

        super().__init__()
        # Minimum velocity of water.
        self.water_vmin = 1470
        # Maximum velocity of water.
        self.water_vmax = 1530
        # Mean water depth (m).
        self.water_dmin = 1000
        # Maximum amplitude of water depth variations.
        self.water_dmax = 5000

    def build_stratigraphy(self):

        vp = Property(name="vp", vmin=self.water_vmin, vmax=self.water_vmax,
                      dzmax=0)
        vs = Property(name="vs", vmin=0, vmax=0)
        rho = Property(name="rho", vmin=2000, vmax=2000)
        water = Lithology(name='water', properties=[vp, vs, rho])
        vp = Property(name="vp", vmin=self.vp_min, vmax=self.vp_max,
                      texture=self.max_texture, trend_min=self.vp_trend_min,
                      trend_max=self.vp_trend_max)
        roc = Lithology(name='roc', properties=[vp, vs, rho])
        if self.amp_max > 0 and self.max_deform_nfreq > 0:
            deform = Deformation(max_deform_freq=self.max_deform_freq,
                                 min_deform_freq=self.min_deform_freq,
                                 amp_max=self.amp_max,
                                 max_deform_nfreq=self.max_deform_nfreq,
                                 prob_deform_change=self.prob_deform_change)
        else:
            deform = None
        waterseq = Sequence(lithologies=[water], ordered=False,
                            thick_min=int(self.water_dmin/self.dh),
                            thick_max=int(self.water_dmax/self.dh))
        rocseq = Sequence(lithologies=[roc], ordered=False, deform=deform)
        strati = Stratigraphy(sequences=[waterseq, rocseq])

        return strati


class PermafrostModelGenerator(BaseModelGenerator):
    def __init__(self):
        super().__init__()

    def build_stratigraphy(self):
        lithologies = {}

        name = "Water"
        vp = Property("vp", vmin=1430, vmax=1430)
        # vs = Property("vs", vmin=0, vmax=0)
        vpvs = Property("vpvs",vmin=0,vmax=0)
        rho = Property("rho", vmin=1000, vmax=1000)
        q = Property("q", vmin=1000, vmax=1000)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen sediments"  # Buckingham 1996, Fig 11
        vp = Property("vp", vmin=1700, vmax=1700, texture=200)
        # vs = Property("vs", vmin=400, vmax=400, texture=150)
        vpvs = Property("vpvs",vmin=4.25, vmax=4.25, texture=1.52)
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

        name = "Frozen Silts"  # Dou 2016, Fig 9, #Buckingham 1996, Fig 11
        vp = Property("vp", vmin=3400, vmax=3400, texture=300)
        # vs = Property("vs", vmin=1888, vmax=1888, texture=170)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.29)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=45, vmax=45, texture=31.5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Partially Frozen Silts"  # Dou 2016, Fig 9, #Buckingham 1996, Fig 11
        vp = Property("vp", vmin=2200, vmax=2200, texture=450)
        # vs = Property("vs", vmin=792, vmax=792, texture=160)
        vpvs = Property("vpvs", vmin=2.78, vmax=2.78, texture=0.94)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=20, vmax=20, texture=10)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Partially Frozen Silts2"  # Dou 2016, Fig 9, #Buckingham 1996, Fig 11
        vp = Property("vp", vmin=1950, vmax=1950, texture=550)
        # vs = Property("vs", vmin=650, vmax=650, texture=186)
        vpvs = Property("vpvs", vmin=3, vmax=3, texture=1.3)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=5)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Shale"  # Bellefleur 2007, Figure 3 zone 2
        vp = Property("vp", vmin=3000, vmax=3000, texture=950)  # IOE Taglu D-43
        # vs = Property("vs", vmin=1666, vmax=650, texture=527)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.87)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=100, vmax=100, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Iperk"  # Bellefleur 2007, Figure 3 zone 2
        vp = Property("vp", vmin=4000, vmax=4000, texture=1500)  # IOE Taglu D-43
        # vs = Property("vs", vmin=2222, vmax=2222, texture=52)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.7)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=100, vmax=100, texture=30)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Unfrozen Shale"  # Bellefleur 2007, Figure 3 zone 2
        vp = Property("vp", vmin=2200, vmax=2200, texture=200)  # IOE Taglu D-43
        # vs = Property("vs", vmin=1222, vmax=1222, texture=111)
        vpvs = Property("vpvs", vmin=1.8, vmax=1.8, texture=0.3)
        rho = Property("rho", vmin=2300, vmax=2300, texture=175)  # king, 1976
        q = Property("q", vmin=70, vmax=100, texture=20)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Frozen Sands2"  # Modified from Matsushima 2016, fig 13c @ 6C Dou 2016
        vp = Property("vp", vmin=2600, vmax=2600, texture=300)
        # vs = Property("vs", vmin=1000, vmax=1000, texture=150)
        vpvs = Property("vpvs", vmin=2.6, vmax=2.6, texture=0.6)
        rho = Property("rho", vmin=1900, vmax=1900, texture=150)
        q = Property("q", vmin=25, vmax=25, texture=10)
        # lithologies[name] = Lithology(name=name, properties=[vp, vs, rho, q])
        lithologies[name] = Lithology(name=name, properties=[vp, vpvs, rho, q])

        name = "Hydrates"  # Modified from Partially frozen Silts Dou 2016, Fig 9, #Buckingham 1996, Fig 11
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
        unfrozen = Sequence(lithologies = [lithologies["Unfrozen sediments"]],
                            deform=deform)
        Permafrost = Sequence(lithologies=[lithologies["Partially Frozen Silts"],
                                           lithologies["Frozen Sands2"],
                                           lithologies["Partially Frozen Silts"],
                                           lithologies["Unfrozen sediments"],
                                           lithologies["Hydrates"]
                                           ],
                              ordered=False, deform=deform)

        strati = Stratigraphy(sequences = [water,unfrozen,Permafrost,unfrozen])

        # sequence = Sequence(lithologies=[lithologies["Water"],
        #                                  lithologies["Unfrozen sediments"],
        #                                  lithologies["Partially Frozen Silts"],
        #                                  lithologies["Frozen Sands2"],
        #                                  lithologies["Partially Frozen Silts"],
        #                                  lithologies["Unfrozen sediments"],
        #                                  lithologies["Hydrates"],
        #                                  lithologies["Unfrozen sediments"]
        #                                  ],
        #                     ordered=True, deform=deform)
        #
        # strati = Stratigraphy(sequences=[sequence])


        return strati

    def generate_model(self, seed=None):

        props2D, layerids, layers = super().generate_model(seed=seed)
        tempvpvs = props2D["vpvs"]
        tempvp = props2D["vp"]
        tempvs = tempvp*0
        tempvs[tempvpvs!=0] = tempvp[tempvpvs!=0] / tempvpvs[tempvpvs!=0]
        props2D["vs"] = tempvs
        # props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers


class MaswModelGenerator(BaseModelGenerator):

    def __init__(self):

        super().__init__()

    def build_stratigraphy(self):

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
                             amp_max=8,
                             max_deform_nfreq=40,
                             prob_deform_change=0.1)

        unsat_seq = Sequence(name="unsaturated",
                             lithologies=[unsaturated_sand],
                             thick_max=25)
        sat_seq = Sequence(name="saturated",
                           lithologies=[saturated_clay,
                                        saturated_sand],
                           thick_max=100)
        weathered_seq = Sequence(name="weathered",
                                 lithologies=[weathered_shale],
                                 thick_max=50)
        roc_seq = Sequence(name="roc",
                           lithologies=[shale],
                           thick_max=99999)

        strati = Stratigraphy(sequences=[unsat_seq,
                                         sat_seq,
                                         weathered_seq,
                                         roc_seq])

        return strati

    def generate_model(self, seed=3):

        props2D, layerids, layers = super().generate_model(seed=seed)
        # Add vs to dictionnary
        props2D["vs"] = props2D["vp"] / props2D["vpvs"]

        return props2D, layerids, layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modgen",
        type=str,
        default="BaseModelGenerator",
        help="Name of the case to use"
    )
    args, unparsed = parser.parse_known_args()

    modgen = eval(args.modgen)()
    modgen.animated_dataset()
