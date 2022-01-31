# -*- coding: utf-8 -*-
"""
EarthModel provides a library to general layered earth models with flexibility
on the type of properties that can be generated.

See MarineModel for how this can be used.
"""

import argparse

from ModelGenerator import (ModelGenerator, Sequence, Stratigraphy,
                            Deformation, Property, Lithology)


class EarthModel(ModelGenerator):
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
        # Probability that a boundary shape will change between two layers.
        self.prob_deform_change = 0.3
        # Whether or not deformation of consecutive layers are added together.
        self.deform_cumulative = False
        # Add random noise two a layer (% or velocity).
        self.max_texture = 0
        # Range of the filter in x for texture creation.
        self.texture_xrange = 0
        # Range of the filter in z for texture creation.
        self.texture_zrange = 0

        self._properties = None
        self._strati = None

    @property
    def properties(self):
        if self._properties is None:
            self._strati, self._properties = self.build_stratigraphy()
        return self._properties

    @property
    def strati(self):
        if self._strati is None:
            self._strati, self._properties = self.build_stratigraphy()
        return self._strati

    def generate_model(self, seed=None):
        """
        Output the media parameters required for seismic modelling.

        :return:
            props2d: A dictionary of gridded properties' name-values pairs.
            layerids: An array with the layer ID of each cell.
            layers: A list of `Layer` objects.
        """
        props2d, layerids, layers = super().generate_model(self.strati,
                                                           seed=seed)

        return props2d, layerids, layers

    def build_stratigraphy(self):
        """
        Build the stratigraphy object that controls model creation.

        :returns:
            strati: A `Stratigraphy` object.
            properties: A dict of properties with key-values pairs `name:
                        [vmin, vmax]` where `vmin` and `vmax` are the minimum
                        and maximum values that can take a property. Each
                        property returned by generate model should be found in
                        this dictionary.
        """
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
                             prob_deform_change=self.prob_deform_change,
                             cumulative=self.deform_cumulative)
        sequence = Sequence(lithologies=[lith], ordered=False, deform=deform)
        strati = Stratigraphy(sequences=[sequence])
        properties = strati.properties()

        return strati, properties


class MarineModel(EarthModel):
    def __init__(self):
        # Minimum water depth (m).
        self.water_dmin = 1000
        # Maximum water depth (m).
        self.water_dmax = 5000
        # Minimum velocity of water.
        self.water_vmin = 1470
        # Maximum velocity of water.
        self.water_vmax = 1530

        super().__init__()

    def build_stratigraphy(self):
        self.thick0min = int(self.water_dmin/self.dh)
        self.thick0max = int(self.water_dmax/self.dh)

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
                            thick_min=self.thick0min,
                            thick_max=self.thick0max,
                            nmin=1)
        self.thick0min = int(self.water_dmin / self.dh)
        self.thick0max = int(self.water_dmax / self.dh)
        rocseq = Sequence(lithologies=[roc], ordered=False, deform=deform)
        strati = Stratigraphy(sequences=[waterseq, rocseq])
        properties = strati.properties()

        return strati, properties



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modgen",
        type=str,
        default="EarthModel",
        help="Name of the dataset to use"
    )
    args, unparsed = parser.parse_known_args()

    modgen = eval(args.modgen)()
    modgen.animate()
