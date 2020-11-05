
import numpy as np
from vrmslearn.SeismicGenerator import Acquisition
from vrmslearn.VelocityModelGenerator import BaseModelGenerator
from vrmslearn.SeismicUtilities import (random_noise,
                                        random_time_scaling,
                                        random_static,
                                        top_mute,
                                        mute_nearoffset,
                                        sortcmp,
                                        smooth_velocity_wavelength,
                                        generate_reflections_ttime,
                                        vdepth2time,
                                        calculate_vrms)


class LabelGenerator:
    """
    A class to generate the label from seismic data and model
    """

    def __init__(self, model: BaseModelGenerator, acquire: Acquisition):
        """

        :param acquire: An Acquisition object describing the seismic acquisition
        :param model: A VelocityModelGenerator describing model creation
        """
        self.acquire = acquire
        self.model = model
        self.label_names = ('ref', 'vrms', 'vint', 'vdepth')
        self.weight_names = ['tweight', 'dweight']

        # The direct arrival is contained in labels.
        self.identify_direct = True
        # If true, mute direct arrival.
        self.mute_dir = False
        # If true, apply random static to the data.
        self.mask_firstvel = False
        self.random_static = False
        # Maximum static in nb of samples.
        self.random_static_max = 2
        # If true, add random noise to the data.
        self.random_noise = False
        # Maximum noise relative to data maximum.
        self.random_noise_max = 0.1
        # If true, mute random near offset traces.
        self.mute_nearoffset = False
        # Maximum offset that can be mutes.
        self.mute_nearoffset_max = 10
        # If true, apply a random gain.
        self.random_time_scaling = False

        # Model gaussian smoothing.
        # See smooth_velocity_wavelength in velocity_transformations.
        # Standard deviation in x.
        self.model_smooth_x = 0
        # Standard deviation in t (z smoothing).
        self.model_smooth_t = 0
        # Train on True:  shot gathers, False: CMP.
        self.train_on_shots = False

    def generate_labels(self, props):
        """
        Output the labels attached to modelling of a particular dataset. In
        this case, we want to predict vp in depth from cmp gathers.

        :param props: A dict with {name_of_prop: array of property}
        :return: labels A list of labels
                 weights A list of weights
        """

        vp, vs, rho = (props["vp"], props["vs"], props["rho"])
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

        vint = np.zeros((self.acquire.NT, vp.shape[1]))
        z0 = int(self.acquire.source_depth / self.model.dh)
        t = np.arange(0, self.acquire.NT, 1) * self.acquire.dt
        for ii in range(vp.shape[1]):
            vint[:, ii] = vdepth2time(vp[z0:, ii], self.model.dh, t,
                                      t0=self.acquire.tdelay)
        vint = vint[::self.acquire.resampling, :]

        tweights = vrms * 0 + 1
        dweights = 2 * np.cumsum(self.model.dh / vp, axis=0) + self.acquire.tdelay
        dweights = dweights - 2 * np.sum(self.model.dh / vp[:z0, :], axis=0)
        for ii in range(vp.shape[1]):
            i_t = np.argwhere(refs[:, ii] > 0.1).flatten()[-1]
            tweights[i_t:, ii] = 0
            mask = dweights[:, ii] >= i_t * self.acquire.dt * self.acquire.resampling
            dweights[mask, ii] = 0
            dweights[dweights[:, ii] != 0, ii] = 1

        # Normalize so the labels are between 0 and 1
        vrms = (vrms - self.model.vp_min) / (self.model.vp_max - self.model.vp_min)
        vint = (vint - self.model.vp_min) / (self.model.vp_max - self.model.vp_min)
        vp = (vp - self.model.vp_min) / (self.model.vp_max - self.model.vp_min)

        labels = [refs, vrms, vint, vp]
        weights = [tweights, dweights]

        return labels, weights

    def preprocess(self, data, labels, weights):
        """
        A function to preprocess the data and labels before feeding it to the
        network.

        @params:
        data (numpy.array): Data array
        labels  (list): A list of numpy.array containing the labels
        weights (list): A list of weights to apply to the outputs

        @returns:
        data (numpy.array): The preprocessed data
        labels (list):      The preprocessed label list
        """
        vp = labels[-1]

        # Adding random noises to the data.
        if self.random_time_scaling:
            data = random_time_scaling(data,
                                       self.acquire.dt * self.acquire.resampling)
        if self.mute_dir:
            wind_length = int(2 / self.acquire.peak_freq / self.acquire.dt
                              / self.acquire.resampling)
            s, r = self.acquire.set_rec_src()
            offsets = [r[0, ii]-s[0, r[3, ii]] for ii in range(r.shape[1])]
            data = top_mute(data, vp[0], wind_length, offsets,
                            self.acquire.dt * self.acquire.resampling,
                            self.acquire.tdelay)
        if self.random_static:
            data = random_static(data, self.random_static_max)
        if self.random_noise:
            data = random_noise(data, self.random_noise_max)
        if self.mute_nearoffset:
            data = mute_nearoffset(data, self.mute_nearoffset_max)

        # Resort the data according to CMP.
        src_pos_all, rec_pos_all = self.acquire.set_rec_src()
        if not self.train_on_shots:
            data, datapos = sortcmp(data, src_pos_all, rec_pos_all)
        else:
            data = np.reshape(data, [data.shape[0], src_pos_all.shape[1], -1])
            data = data.swapaxes(1, 2)
            datapos = src_pos_all[0, :]

        # Smooth the velocity model.
        if self.model_smooth_x != 0 or self.model_smooth_t != 0:
            labels[-1] = smooth_velocity_wavelength(labels[-1], self.model.dh,
                                                    self.model_smooth_t,
                                                    self.model_smooth_x)

        # Resample labels in x to correspond to data position.
        x = np.arange(0, self.model.NX) * self.model.dh
        ind1 = np.argmax(x >= datapos[0])
        ind2 = -np.argmax(x[::-1] <= datapos[-1])

        labels = [label[:, ind1:ind2:self.acquire.ds] for label in labels]
        weights = [w[:, ind1:ind2:self.acquire.ds] for w in weights]

        for label in labels:
            if label.shape[-1] != data.shape[-1]:
                raise ValueError("Number of x positions in label and number cmp"
                                 " mismatch.")

        # We can predict velocities under the source and receiver arrays only.
        sz = int(self.acquire.source_depth / self.model.dh)

        labels[-1] = labels[-1][sz:, :]
        weights[-1] = weights[-1][sz:, :]

        data = np.expand_dims(data, axis=-1)
        # labels = [np.expand_dims(label, axis=-1) for label in labels]
        # weights = [np.expand_dims(weight, axis=-1) for weight in weights]

        return data, labels, weights

    def postprocess(self, labels, preds, vproc=True):
        """
        A function to postprocess the predictions.

        @params:
        labels  (dict): A dict containing {labelname: label}
        preds  (dict): A dict containing {predname: prediction}

        @returns:
        labels (dict):      The preprocessed labels {labelname: processed_label}
        preds (dict):       The preprocessed predictions
        """
        if vproc:
            for el in ['vrms', 'vint', 'vdepth']:
                if el in labels:
                    labels[el] = labels[el] * (self.model.vp_max -
                                          self.model.vp_min) + self.model.vp_min
                if el in preds:
                    preds[el] = preds[el] * (self.model.vp_max -
                                 self.model.vp_min) + self.model.vp_min
        if 'ref' in preds:
            preds['ref'] = np.argmax(preds['ref'], axis=2)

        return labels, preds


class PermafrostLabelGenerator(LabelGenerator):
    def __init__(self,model: BaseModelGenerator, acquire: Acquisition):
        super().__init__(model,acquire)

    def generate_labels(self, props):
        """
        Reduced labels to vp and vs only
        Output the labels attached to modelling of a particular dataset. In
        this case, we want to predict vp in depth from cmp gathers.

        :param props: A dict with {name_of_prop: array of property}
        :return: labels A list of labels
                 weights A list of weights
        """

        vp, vs, rho = (props["vp"], props["vs"], props["rho"])
        vp = (vp - self.model.vp_min) / (self.model.vp_max - self.model.vp_min)
        vs = (vs - vs.min())/(vs.max()-vs.min())

        tweights = vp*0 + 1
        dweights = vp*0 + 1
        labels = [vp,vs]
        weights = [tweights,dweights]

        return labels, weights

    def preprocess(self, data, labels, weights):
        """
        Removed resorted data to CMP
        A function to preprocess the data and labels before feeding it to the
        network.

        @params:
        data (numpy.array): Data array
        labels  (list): A list of numpy.array containing the labels
        weights (list): A list of weights to apply to the outputs

        @returns:
        data (numpy.array): The preprocessed data
        labels (list):      The preprocessed label list
        """
        vp = labels[-1]

        # Adding random noises to the data.
        if self.random_time_scaling:
            data = random_time_scaling(data,
                                       self.acquire.dt * self.acquire.resampling)
        if self.mute_dir:
            wind_length = int(2 / self.acquire.peak_freq / self.acquire.dt
                              / self.acquire.resampling)
            s, r = self.acquire.set_rec_src()
            offsets = [r[0, ii]-s[0, r[3, ii]] for ii in range(r.shape[1])]
            data = top_mute(data, vp[0], wind_length, offsets,
                            self.acquire.dt * self.acquire.resampling,
                            self.acquire.tdelay)
        if self.random_static:
            data = random_static(data, self.random_static_max)
        if self.random_noise:
            data = random_noise(data, self.random_noise_max)
        if self.mute_nearoffset:
            data = mute_nearoffset(data, self.mute_nearoffset_max)

        # # Resort the data according to CMP.
        src_pos_all, rec_pos_all = self.acquire.set_rec_src()
        data = np.reshape(data, [data.shape[0], src_pos_all.shape[1], -1])
        data = data.swapaxes(1, 2)
        datapos = src_pos_all[0, :]

        # Smooth the velocity model.
        if self.model_smooth_x != 0 or self.model_smooth_t != 0:
            labels[-1] = smooth_velocity_wavelength(labels[-1], self.model.dh,
                                                    self.model_smooth_t,
                                                    self.model_smooth_x)

        # Resample labels in x to correspond to data position.
        x = np.arange(0, self.model.NX) * self.model.dh
        ind1 = np.argmax(x >= datapos[0])
        ind2 = -np.argmax(x[::-1] <= datapos[-1])

        labels = [label[:, ind1:ind1+1:2] for label in labels]
        weights = [w[:, ind1:ind1+1:2] for w in weights]

        for label in labels:
            if label.shape[-1] != data.shape[-1]:
                raise ValueError("Number of x positions in label and number cmp"
                                 " mismatch.")

        # We can predict velocities under the source and receiver arrays only.
        sz = int(self.acquire.source_depth / self.model.dh)

        labels[-1] = labels[-1][sz:, :]
        weights[-1] = weights[-1][sz:, :]

        data = np.expand_dims(data, axis=-1)

        return data, labels, weights