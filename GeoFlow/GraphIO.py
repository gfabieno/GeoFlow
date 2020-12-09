import numpy as np
from matplotlib import pyplot as plt

from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.EarthModel import EarthModel
from GeoFlow.SeismicUtilities import (smooth_velocity_wavelength,
                                      generate_reflections_ttime,
                                      vdepth2time,
                                      calculate_vrms,
                                      random_noise,
                                      random_time_scaling,
                                      random_static,
                                      top_mute,
                                      mute_nearoffset,
                                      sortcmp,
                                      dispersion_curve)


class GraphOutput:
    """
    Generate the output, label and weight of a network.

    :param naxes: The quantity of figures required by this output.
    :type naxes: int
    """
    name = "Baseoutput"
    naxes = 1

    def __init__(self, model: EarthModel, acquire: Acquisition):
        """
        Define the input data format, as implied by `model` and `acquire`.

        Additional attributes that control how an output is generated can be
        defined here.

        :param acquire: An `Acquisition` object describing seismic acquisition.
        :type acquire: Acquisition
        :param model: A `VelocityModelGenerator` describing model creation.
        :type model: EarthModel
        """
        self.acquire = acquire
        self.model = model

    def plot(self, data, ax=None, cmap=plt.get_cmap('hot'), vmin=0,
             vmax=1, clip=1, im=None):
        """
        Plot the output.

        :param data: The data to plot.
        :param ax: The axis on which to plot.
        :param cmap: The colormap.
        :param vmin: Minimum value of the colormap. If None, defaults to
                     `-clip * max` of data.
        :param vmax: Maximum value of the colormap. If None, defaults to
                     `clip * max` of data.
        :param clip: Clipping of the data.
        :param im: If provided, the image data is updated.

        :return:
            im: Return value of `ax.imshow`.
        """
        if vmax is None:
            vmax = np.max(data) * clip
        if vmin is None:
            vmin = -vmax

        data = np.reshape(data, [data.shape[0], -1])
        if im is None:
            im = ax.imshow(data,
                           interpolation='bilinear',
                           cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           aspect='auto')
            ax.set_title("Output: %s" % self.name,
                         fontsize=16, fontweight='bold')
            _ = ax.get_position().get_points().flatten()
            plt.colorbar(im, ax=ax)
        else:
            im.set_array(data)

        return im

    def generate(self, props):
        """
        Output the labels and weights from a dict of earth properties.

        :param props: A dictionary of properties' name-values pairs.

        :return:
            labels: An array of labels.
            weights: An array of weights. When computing the loss, the outputs
                     and labels are multiplied by weight.
        """
        raise NotImplementedError

    def preprocess(self, label, weight):
        """
        Preprocess the data and labels before feeding it to the network.

        :param labels: An array containing the labels.
        :param weights: An array containing the weight. When computing the
                        loss, the outputs and labels are multiplied by weight.

        :return:
            data: The preprocessed data ready to be fed to the network.
        """
        raise NotImplementedError

    def postprocess(self, label):
        """
        Postprocess the outputs.

        :param label: The output to postprocess.

        :return:
            labels: The preprocessed output.
        """
        raise NotImplementedError


class Reftime(GraphOutput):
    name = "ref"

    def __init__(self, model: EarthModel, acquire: Acquisition):
        super().__init__(model, acquire)
        self.identify_direct = False
        self.train_on_shots = False

    def generate(self, props):
        vp, vs, rho = props["vp"], props["vs"], props["rho"]
        refs = np.zeros((self.acquire.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            refs[:, ii] = generate_reflections_ttime(vp[:, ii],
                                                     self.acquire.source_depth,
                                                     self.model.dh,
                                                     self.acquire.NT,
                                                     self.acquire.dt,
                                                     self.acquire.peak_freq,
                                                     self.acquire.tdelay,
                                                     self.acquire.minoffset,
                                                     self.identify_direct)
        refs = refs[::self.acquire.resampling, :]
        return refs, refs * 0 + 1

    def preprocess(self, label, weight):
        src_pos_all, rec_pos_all = self.acquire.set_rec_src()
        if not self.train_on_shots:
            data, datapos = sortcmp(None, src_pos_all, rec_pos_all)
        else:
            datapos = src_pos_all[0, :]
        # Resample labels in x to correspond to data position.
        x = np.arange(0, self.model.NX) * self.model.dh
        ind1 = np.argmax(x >= datapos[0])
        ind2 = -np.argmax(x[::-1] <= datapos[-1])

        label = label[:, ind1:ind2:self.acquire.ds]
        weight = weight[:, ind1:ind2:self.acquire.ds]

        return label, weight

    def postprocess(self, label):
        if len(label.shape) > 2:
            label = np.argmax(label, axis=2)

        return label


class Vrms(Reftime):
    name = "vrms"

    def generate(self, props):
        vp, vs, rho = props["vp"], props["vs"], props["rho"]
        vrms = np.zeros((self.acquire.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            vrms[:, ii] = calculate_vrms(vp[:, ii],
                                         self.model.dh,
                                         self.acquire.Npad,
                                         self.acquire.NT,
                                         self.acquire.dt,
                                         self.acquire.tdelay,
                                         self.acquire.source_depth)

        vrms = vrms[::self.acquire.resampling, :]

        refs = np.zeros((self.acquire.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            refs[:, ii] = generate_reflections_ttime(vp[:, ii],
                                                     self.acquire.source_depth,
                                                     self.model.dh,
                                                     self.acquire.NT,
                                                     self.acquire.dt,
                                                     self.acquire.peak_freq,
                                                     self.acquire.tdelay,
                                                     self.acquire.minoffset,
                                                     self.identify_direct)
        refs = refs[::self.acquire.resampling, :]
        tweights = vrms * 0 + 1
        for ii in range(vp.shape[1]):
            i_t = np.argwhere(refs[:, ii] > 0.1).flatten()[-1]
            tweights[i_t:, ii] = 0

        return vrms, tweights

    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        vmin, vmax = self.model.properties["vp"]
        label = (label - vmin) / (vmax - vmin)
        return label, weight

    def postprocess(self, label):
        vmin, vmax = self.model.properties["vp"]
        return label * (vmax - vmin) + vmin


class Vint(Vrms):
    name = "vint"

    def generate(self, props):
        vp, vs, rho = props["vp"], props["vs"], props["rho"]
        vint = np.zeros((self.acquire.NT, vp.shape[1]))
        z0 = int(self.acquire.source_depth / self.model.dh)
        t = np.arange(0, self.acquire.NT, 1) * self.acquire.dt
        for ii in range(vp.shape[1]):
            vint[:, ii] = vdepth2time(vp[z0:, ii], self.model.dh, t,
                                      t0=self.acquire.tdelay)
        vint = vint[::self.acquire.resampling, :]

        refs = np.zeros((self.acquire.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            refs[:, ii] = generate_reflections_ttime(vp[:, ii],
                                                     self.acquire.source_depth,
                                                     self.model.dh,
                                                     self.acquire.NT,
                                                     self.acquire.dt,
                                                     self.acquire.peak_freq,
                                                     self.acquire.tdelay,
                                                     self.acquire.minoffset,
                                                     self.identify_direct)
        refs = refs[::self.acquire.resampling, :]
        tweights = vint * 0 + 1
        for ii in range(vp.shape[1]):
            i_t = np.argwhere(refs[:, ii] > 0.1).flatten()[-1]
            tweights[i_t:, ii] = 0

        return vint, tweights


class Vdepth(Vrms):
    name = "vdepth"

    def __init__(self, model: EarthModel, acquire: Acquisition):
        super().__init__(model, acquire)
        self.train_on_shots = False
        self.model_smooth_t = 0
        self.model_smooth_x = 0

    def generate(self, props):
        vp, vs, rho = props["vp"], props["vs"], props["rho"]
        z0 = int(self.acquire.source_depth / self.model.dh)
        refs = np.zeros((self.acquire.NT, vp.shape[1]))
        for ii in range(vp.shape[1]):
            refs[:, ii] = generate_reflections_ttime(vp[:, ii],
                                                     self.acquire.source_depth,
                                                     self.model.dh,
                                                     self.acquire.NT,
                                                     self.acquire.dt,
                                                     self.acquire.peak_freq,
                                                     self.acquire.tdelay,
                                                     self.acquire.minoffset,
                                                     self.identify_direct)
        refs = refs[::self.acquire.resampling, :]
        dweights = 2 * np.cumsum(self.model.dh / vp,
                                 axis=0) + self.acquire.tdelay
        dweights = dweights - 2 * np.sum(self.model.dh / vp[:z0, :], axis=0)
        for ii in range(vp.shape[1]):
            i_t = np.argwhere(refs[:, ii] > 0.1).flatten()[-1]
            threshold = i_t * self.acquire.dt * self.acquire.resampling
            mask = dweights[:, ii] >= threshold
            dweights[mask, ii] = 0
            dweights[dweights[:, ii] != 0, ii] = 1

        return vp, dweights

    def preprocess(self, label, weight):
        # Smooth the velocity model.
        if self.model_smooth_x != 0 or self.model_smooth_t != 0:
            label = smooth_velocity_wavelength(label,
                                               self.model.dh,
                                               self.model_smooth_t,
                                               self.model_smooth_x)

        label, weight = super().preprocess(label, weight)
        # We can predict velocities under the source and receiver arrays only.
        sz = int(self.acquire.source_depth / self.model.dh)
        label = label[sz:, :]
        weight = weight[sz:, :]

        return label, weight


class Vsdepth(Reftime):
    name = "vsdepth"

    def generate(self, props):
        vp, vs, rho = props["vp"], props["vs"], props["rho"]
        return vs, vs * 0 + 1

    def preprocess(self, label, weight):
        # TODO find a way to get vs min and max
        label, weight = super().preprocess(label, weight)
        vmin, vmax = self.model.properties["vs"]
        label = (label - vmin) / (vmax - vmin)
        return label, weight

    def postprocess(self, label):
        vmin, vmax = self.model.properties["vs"]
        return label * (vmax - vmin) + vmin


# TODO Fix the plotting as it was before for multiple shot case (2D)
class GraphInput:
    name = "BaseInput"
    naxes = 1

    def __init__(self, acquire: Acquisition, model: EarthModel):
        self.acquire = acquire
        self.model = model

    def plot(self, data, ax, cmap=plt.get_cmap('Greys'), vmin=None, vmax=None,
             clip=0.1, im=None):
        """
        Plot this input using default values.

        :param data: The data to plot.
        :param ax: The axis on which to plot.
        :param cmap: The colormap.
        :param vmin: Minimum value of the colormap. If None, default to
                     -clip * max of data.
        :param vmax: Maximum value of the colormap. If None, default to
                     clip * max of data.
        :param clip: Clipping of the data.
        :param im: If provided, the image data is updated.

        :return:
           im:     return value of ax.imshow
       """
        if vmax is None:
            vmax = np.max(data) * clip
        if vmin is None:
            vmin = -vmax

        data = np.reshape(data, [data.shape[0], -1])
        if im is None:
            im = ax.imshow(data,
                           interpolation='bilinear',
                           cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           aspect='auto')
            ax.set_title("Input: %s" % self.name,
                         fontsize=16, fontweight='bold')
        else:
            im.set_array(data)

        return im

    def generate(self, data):
        """
        Compute the graph input from the modeled data to be saved on disk.
        """
        return data

    def preprocess(self, data, labels):
        """
        Preprocess the data before feeding it to the network.

        :param data: The input data.
        :param labels: A dictionary of labels' name-value pairs.

        :param data: The preprocessed data ready to be fed to the network.
       """
        return data


# TODO Jefferson complete that
class Dispersion(GraphInput):
    name = "dispersion"

    def __init__(self, acquire: Acquisition, model: EarthModel, cmax , cmin):
        self.acquire = acquire
        self.model = model
        self.cmax, self.cmin = cmax, cmin

    def generate(self, data):
        src_pos, rec_pos = self.acquire.set_rec_src()
        dt = self.acquire.dt
        d = dispersion_curve(data, self.acquire.re)
        return d


class ShotGather(GraphInput):
    name = "shotgather"

    def __init__(self,
                 acquire: Acquisition,
                 model: EarthModel,
                 train_on_shots: bool = False,
                 mute_dir: bool = False,
                 random_static: bool = False,
                 random_static_max: int = 2,
                 random_noise: bool = False,
                 random_noise_max: float = 0.1,
                 mute_nearoffset: bool = False,
                 mute_nearoffset_max: float = 10,
                 random_time_scaling: bool = False):
        """
        Define parameters controlling the preprocessing.

        :param acquire: An `Acquisition` object controlling data creation.
        :param model: A `EarthModel` object.
        :param train_on_shots: If true, the data is in shots order, else, it
                               is sorted by CMP.
        :param mute_dir: If true, mute direct arrival.
        :param random_static: If true, apply random static to the data.
        :param random_static_max: Maximum static in nb of samples.
        :param random_noise: If true, add random noise to the data.
        :param random_noise_max: Maximum noise relative to data maximum.
        :param mute_nearoffset: If true, mute random near offset traces.
        :param mute_nearoffset_max: Maximum offset that can be mutes.
        :param random_time_scaling: If true, apply a random gain in time.
        """
        self.acquire = acquire
        self.model = model
        self.train_on_shots = train_on_shots
        self.mute_dir = mute_dir
        self.random_static = random_static
        self.random_static_max = random_static_max
        self.random_noise = random_noise
        self.random_noise_max = random_noise_max
        self.mute_nearoffset = mute_nearoffset
        self.mute_nearoffset_max = mute_nearoffset_max
        self.random_time_scaling = random_time_scaling

        self.is_1d = self.acquire.singleshot
        self.naxes = 1 if self.is_1d else 2

    # TODO Handle 2D case.
    def preprocess(self, data, labels):
        # Add random noises to the data.
        if self.random_time_scaling:
            dt = self.acquire.dt * self.acquire.resampling
            data = random_time_scaling(data, dt)
        if self.mute_dir:
            vp = labels["vdepth"]
            wind_length = int(2 / self.acquire.peak_freq / self.acquire.dt
                              / self.acquire.resampling)
            s, r = self.acquire.set_rec_src()
            offsets = [r[0, ii] - s[0, r[3, ii]] for ii in range(r.shape[1])]
            data = top_mute(data, vp[0], wind_length, offsets,
                            self.acquire.dt * self.acquire.resampling,
                            self.acquire.tdelay)
        if self.random_static:
            data = random_static(data, self.random_static_max)
        if self.random_noise:
            data = random_noise(data, self.random_noise_max)
        if self.mute_nearoffset:
            data = mute_nearoffset(data, self.mute_nearoffset_max)

        src_pos_all, rec_pos_all = self.acquire.set_rec_src()
        if not self.train_on_shots:
            data, datapos = sortcmp(data, src_pos_all, rec_pos_all)
        else:
            data = np.reshape(data, [data.shape[0], src_pos_all.shape[1], -1])
            data = data.swapaxes(1, 2)

        data = np.expand_dims(data, axis=-1)

        return data
