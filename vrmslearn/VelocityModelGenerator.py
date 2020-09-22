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
        (vp, vs, rho), layerids, layers = super().generate_model(self.strati,
                                                                 seed=seed)
        return (vp, vs, rho), layerids, layers

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
