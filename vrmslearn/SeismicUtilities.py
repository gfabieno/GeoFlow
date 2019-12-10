import numpy as np
from scipy.signal import convolve2d

def gaussian(f0, t, o, amp=1.0, order=2):

    x = np.pi * f0 * (t + o)
    e = amp * np.exp(-x ** 2)
    if order == 1:
        return e*x
    elif order == 2:
        return (1.0 - 2.0 * x ** 2) * e
    elif order == 3:
        return 2.0 * x * (2.0 * x ** 2 - 3.0) * e
    elif order == 4:
        return (-8.0 * x ** 4 + 24.0 * x ** 2 - 6.0) * e
    elif order == 5:
        return 4.0 * x * (4.0 * x ** 2 - 20.0 * x ** 2 + 15.0) * e
    elif order == 6:
        return -4.0 * (8.0 * x ** 6 - 60.0 * x ** 4 + 90.0 * x ** 2 - 15.0) * e

def morlet(f0, t, o , amp=1.0, order=5):
    x = f0 * (t + o)
    return amp * np.cos(x*order) * np.exp(- x ** 2)


def shift_trace(signal, phase):
    S = np.fft.fft(signal)
    NT = len(signal)
    S[1:NT//2] *= 2.0
    S[NT // 2+1:] *=0
    s = np.fft.ifft(S)
    return np.real(s) * np.cos(phase) + np.imag(s) * np.sin(phase)



def random_wavelet_generator(NT, dt, peak_freq, df, tdelay):

    allwavefuns = [lambda f0, t, o: gaussian(f0, t, o, order=1),
                   lambda f0, t, o: gaussian(f0, t, o, order=2),
                   lambda f0, t, o: gaussian(f0, t, o, order=3),
                   lambda f0, t, o: morlet(f0, t, o, order=2),
                   lambda f0, t, o: morlet(f0, t, o, order=3),
                   lambda f0, t, o: morlet(f0, t, o, order=4)]


    def random_wavelet():
        t = np.arange(0, NT) * dt
        fmin = peak_freq - df
        fmax = peak_freq + df
        f0 = np.random.rand(1) * (fmax - fmin) + fmin
        phase = np.random.rand(1) * np.pi
        fun = np.random.choice(allwavefuns)
        src = fun(f0, t, -tdelay)
        src = shift_trace(src, phase)

        return src

    return random_wavelet

def mask_batch(batch,
               mask_fraction,
               mask_time_frac):
    for ii, el in enumerate(batch):
        data = el[0]
        NT = data.shape[0]
        ng = data.shape[1]

        # Mask time and offset
        frac = np.random.rand() * mask_time_frac
        twindow = int(frac * NT)
        owindow = int(frac * ng / 2)
        batch[ii][0][-twindow:, :] = 0
        batch[ii][0][:, :owindow] = 0
        batch[ii][0][:, -owindow:] = 0

        # take random subset of traces
        ntokill = int(np.random.rand() * mask_fraction * ng * frac)
        tokill = np.random.choice(np.arange(owindow, ng - owindow), ntokill,
                                  replace=False)
        batch[ii][0][:, tokill] = 0

        batch[ii][3][-twindow:] = 0

    return batch


def mute_direct(data, vp0, pars, offsets=None):
    wind_length = int(2 / pars.peak_freq / pars.dt / pars.resampling)
    taper = np.arange(wind_length)
    taper = np.sin(np.pi * taper / (2 * wind_length - 1)) ** 2
    NT = data.shape[0]
    ng = data.shape[1]
    if offsets is None:
        if pars.gmin is None or pars.gmax is None:
            offsets = (np.arange(0, ng) - (ng) / 2) * pars.dh * pars.dg
        else:
            offsets = (np.arange(pars.gmin, pars.gmax, pars.dg)) * pars.dh

    for ii, off in enumerate(offsets):
        tmute = int(
            (np.abs(off) / vp0 + 1.5 * pars.tdelay) / pars.dt / pars.resampling)
        if tmute <= NT:
            data[0:tmute, ii] = 0
            mute_max = np.min([tmute + wind_length, NT])
            nmute = mute_max - tmute
            data[tmute:mute_max, ii] = data[tmute:mute_max, ii] * taper[:nmute]
        else:
            data[:, ii] = 0

    return data


def random_static(data, max_static):
    ng = data.shape[1]
    shifts = (np.random.rand(ng) - 0.5) * max_static * 2
    for ii in range(ng):
        data[:, ii] = np.roll(data[:, ii], int(shifts[ii]), 0)
    return data


def random_noise(data, max_amp):
    max_amp = max_amp * np.max(data) * 2.0
    data = data + (np.random.rand(data.shape[0], data.shape[1]) - 0.5) * max_amp
    return data


def mute_nearoffset(data, max_off):
    data[:, :np.random.randint(max_off)] *= 0
    return data


def random_filt(data, filt_length):
    filt_length = int((np.random.randint(filt_length) // 2) * 2 + 1)
    filt = np.random.rand(filt_length, 1)
    data = convolve2d(data, filt, 'same')
    return data


def random_time_scaling(data, dt, emin=-2.0, emax=2.0, scalmax=None):
    t = np.reshape(np.arange(0, data.shape[0]) * dt, [data.shape[0], 1])
    e = np.random.rand() * (emax - emin) + emin
    scal = (t + 1e-6) ** e
    if scalmax is not None:
        scal[scal > scalmax] = scalmax
    return data * scal

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

def generate_reflections_ttime(vp,
                               pars,
                               tol=0.015,
                               window_width=0.45):
    """
    Output the reflection travel time at the minimum offset of a CMP gather

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model
    tol (float): The minimum relative velocity change to consider a reflection
    window_width (float): time window width in percentage of pars.peak_freq

    @returns:

    tabel (numpy.ndarray) : A 2D array with pars.NT elements with 1 at reflecion
                            times +- window_width/pars.peak_freq, 0 elsewhere
    """

    vp = vp[int(pars.source_depth / pars.dh):]
    vlast = vp[0]
    ind = []
    for ii, v in enumerate(vp):
        if np.abs((v - vlast) / vlast) > tol:
            ind.append(ii - 1)
            vlast = v

    if pars.minoffset != 0:
        dt = 2.0 * pars.dh / vp
        t0 = np.cumsum(dt)
        vrms = np.sqrt(t0 * np.cumsum(vp ** 2 * dt))
        tref = np.sqrt(
            t0[ind] ** 2 + pars.minoffset ** 2 / vrms[ind] ** 2) + pars.tdelay
    else:
        ttime = 2 * np.cumsum(pars.dh / vp) + pars.tdelay
        tref = ttime[ind]

    if pars.identify_direct:
        dt = 0
        if pars.minoffset != 0:
            dt = pars.minoffset / vp[0]
        tref = np.insert(tref, 0, pars.tdelay + dt)

    tlabel = np.zeros(pars.NT)
    for t in tref:
        imin = int(t / pars.dt - window_width / pars.peak_freq / pars.dt)
        imax = int(t / pars.dt + window_width / pars.peak_freq / pars.dt)
        if imin <= pars.NT and imax <= pars.NT:
            tlabel[imin:imax] = 1

    return tlabel


def two_way_travel_time(vp, pars):
    """
    Output the two-way travel-time for each cell in vp

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:

    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth, cut to
                          have the same size of t
    t (numpy.ndarray) :  The two-way travel time of each cell

    """
    vpt = vp[int(pars.source_depth / pars.dh):]
    t = 2 * np.cumsum(pars.dh / vpt) + pars.tdelay
    t = t[t < pars.NT * pars.dt]
    vpt = vpt[:len(t)]

    return vpt, t


def interval_velocity_time(vp, pars):
    """
    Output the interval velocity in time

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:

    vint (numpy.ndarray) : The interval velocity in time

    """
    vpt, t = two_way_travel_time(vp, pars)
    interpolator = interp1d(t, vpt,
                            bounds_error=False,
                            fill_value="extrapolate",
                            kind="nearest")
    vint = interpolator(np.arange(0, pars.NT, 1) * pars.dt)

    return vint

def interval_velocity_depth(vp, vint, pars):
    """
    Output the interval velocity in time

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:

    vint (numpy.ndarray) : The interval velocity in time

    """
    #vpt, t = two_way_travel_time(vp, pars)
    t = np.arange(0, pars.NT, 1) * pars.dt
    interpolator = interp1d(t*vint/2.0 + pars.source_depth, vint,
                            bounds_error=False,
                            fill_value="extrapolate",
                            kind="linear")
    vdepth = interpolator(np.arange(0, pars.NZ, 1) * pars.dh)

    return vdepth

def calculate_vrms(vp, dh, Npad, NT, dt, tdelay, source_depth):
    """
    This method inputs vp and outputs the vrms. The global parameters in
    common.py are used for defining the depth spacing, source and receiver
    depth etc. This method assumes that source and receiver depths are same.

    The convention used is that the velocity denoted by the interval
    (i, i+1) grid points is given by the constant vp[i+1].

    @params:
    vp (numpy.ndarray) :  1D vp values in meters/sec.
    dh (float) : the spatial grid size
    Npad (int) : Number of absorbing padding grid points over the source
    NT (int)   : Number of time steps of output
    dt (float) : Time step of the output
    tdelay (float): Time before source peak
    source_depth (float) The source depth in meters


    @returns:
    vrms (numpy.ndarray) : numpy array of shape (NT, ) with vrms
                           values in meters/sec.
    """

    NZ = vp.shape[0]

    # Create a numpy array of depths corresponding to the vp grid locations
    depth = np.zeros((NZ,))
    for i in range(NZ):
        depth[i] = i * dh

    # Create a list of tuples of (relative depths, velocity) of the layers
    # following the depth of the source / receiver depths, till the last layer
    # before the padding zone at the bottom
    last_depth = dh * (NZ - Npad - 1)
    rdepth_vel_pairs = [(d - source_depth, vp[i]) for i, d in enumerate(depth)
                        if d > source_depth and d <= last_depth]
    first_layer_vel = rdepth_vel_pairs[0][1]
    rdepth_vel_pairs.insert(0, (0.0, first_layer_vel))

    # Calculate a list of two-way travel times
    t = [2.0 * (rdepth_vel_pairs[index][0] - rdepth_vel_pairs[index - 1][
        0]) / vel
         for index, (_, vel) in enumerate(rdepth_vel_pairs) if index > 0]
    t.insert(0, 0.0)
    total_time = 0.0
    for i, time in enumerate(t):
        total_time += time
        t[i] = total_time

    # The last time must be 'dt' * 'NT', so adjust the lists 'rdepth_vel_pairs'
    # and 't' by cropping and adjusting the last sample accordingly
    rdepth_vel_pairs = [(rdepth_vel_pairs[i][0], rdepth_vel_pairs[i][1]) for
                        i, time in enumerate(t)
                        if time <= NT * dt]
    t = [time for time in t if time <= NT * dt]
    last_index = len(t) - 1
    extra_distance = (NT * dt - t[last_index]) * rdepth_vel_pairs[last_index][
        1] / 2.0
    rdepth_vel_pairs[last_index] = (
        extra_distance + rdepth_vel_pairs[last_index][0],
        rdepth_vel_pairs[last_index][1])
    t[last_index] = NT * dt

    # Compute vrms at the times in t
    vrms = [first_layer_vel]
    sum_numerator = 0.0
    for i in range(1, len(t)):
        sum_numerator += (t[i] - t[i - 1]) * rdepth_vel_pairs[i][1] * \
                         rdepth_vel_pairs[i][1]
        vrms.append((sum_numerator / t[i]) ** 0.5)

    # Interpolate vrms to uniform time grid
    tgrid = np.asarray(range(0, NT)) * dt
    vrms = np.interp(tgrid, t, vrms)
    vrms = np.reshape(vrms, [-1])
    # Adjust for time delay
    t0 = int(tdelay / dt)
    vrms[t0:] = vrms[:-t0]
    vrms[:t0] = vrms[t0]

    # Return vrms
    return vrms

def smooth_velocity_wavelength(vp, dh, lt, lx):

    vdepth = vp * 0
    dt = dh / np.max(vp) / 10.0
    NT = int(np.max(2 * np.cumsum(dh / vp, axis=0)) / dt*1)
    t = np.arange(0, NT, 1) * dt
    vint = np.zeros([NT, vp.shape[1]])
    for ii in range(0, vp.shape[1]):
        ti = 2 * np.cumsum(dh / vp[:,ii])
        ti = ti - ti[0]
        interpolator = interp1d(ti, vp[:, ii],
                                bounds_error=False,
                                fill_value="extrapolate",
                                kind="nearest")
        vint[:, ii] = interpolator(t)

    if lt > 0:
        vint = gaussian_filter(vint, [lt//dt, lx])

    for ii in range(0, vp.shape[1]):

        vavg = np.cumsum(dt*vint[:, ii])/2/t
        vavg[0] = vint[0, ii]
        interpolator = interp1d(t * vavg, vint[:, ii],
                                bounds_error=False,
                                fill_value="extrapolate",
                                kind="nearest")
        vdepth[:, ii] = interpolator(np.arange(0, vp.shape[0], 1) * dh)

    return vdepth
