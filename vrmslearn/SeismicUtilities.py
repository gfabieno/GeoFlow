"""
Functions to handle seismic data and velocity models.
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d, CubicSpline


def gaussian(f0, t, o, amp=1.0, order=2):
    """
    Generate a gaussian wavelet

    @params:
    f0 (float): Center frequency of the wavelet
    t  (numpy.array): Time vector
    o   (float): Zero time of the wavelet
    amp (float): Maximum amplitude of the wavelet
    order (int): Wavelet is given by the nth order derivative of a Gaussian

    @returns:
    (numpy.array): The wavelet signal
    """
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


def morlet(f0, t, o, amp=1.0, order=5):
    """
    Generate a morlet wavelet

    @params:
    f0 (float): Center frequency of the wavelet
    t  (numpy.array): Time vector
    o   (float): Zero time of the wavelet
    amp (float): Maximum amplitude of the wavelet
    order (int): Order of the morlet wavelet

    @returns:
    (numpy.array): The wavelet signal
    """
    x = f0 * (t + o)
    return amp * np.cos(x*order) * np.exp(- x ** 2)


def shift_trace(signal, phase):
    """
    Apply a phase rotation to a wavelet

    @params:
    signal  (numpy.array): The wavelet to rotate
    phase   (float): The phase of the rotation

    @returns:
    (numpy.array): The rotated wavelet signal
    """
    S = np.fft.fft(signal)
    nt = len(signal)
    S[1:nt//2] *= 2.0
    S[nt//2+1:] *= 0
    s = np.fft.ifft(S)
    return np.real(s) * np.cos(phase) + np.imag(s) * np.sin(phase)


def random_wavelet_generator(nt, dt, peak_freq, df, tdelay, shapes=(1,)):
    """
    Generate function (callable) that output random wavelets.

    @params:
    nt  (float): Number of time steps
    dt   (float): Time step
    peak_freq (float): Mean peak frequency of the wavelets
    df (float): Peak frequency will be peak_freq +- df
    tdelay (float): Time zero of the wavelets
    shapes (list): List containing which wavelet shapes that can be generated:
                    0-2 1-3th order Gaussian wavelet
                    3-5 2-4th order Morlet wavelet

    @returns:
    (callable): A random wavelet generator
    """
    allwavefuns = [
        lambda f0, t, o: gaussian(f0, t, o, order=1),
        lambda f0, t, o: gaussian(f0, t, o, order=2),
        lambda f0, t, o: gaussian(f0, t, o, order=3),
        lambda f0, t, o: morlet(f0, t, o, order=2),
        lambda f0, t, o: morlet(f0, t, o, order=3),
        lambda f0, t, o: morlet(f0, t, o, order=4),
    ]

    allwavefuns = [allwavefuns[ii] for ii in shapes]

    def random_wavelet():
        t = np.arange(0, nt) * dt
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
        nt = data.shape[0]
        ng = data.shape[1]

        # Mask time and offset
        frac = np.random.rand() * mask_time_frac
        twindow = int(frac * nt)
        owindow = int(frac * ng / 2)
        batch[ii][0][-twindow:, :] = 0
        batch[ii][0][:, :owindow] = 0
        batch[ii][0][:, -owindow:] = 0

        # take random subset of traces
        ntokill = int(np.random.rand() * mask_fraction * ng * frac)
        tokill = np.random.choice(
            np.arange(owindow, ng - owindow),
            ntokill,
            replace=False,
        )
        batch[ii][0][:, tokill] = 0

        batch[ii][3][-twindow:] = 0

    return batch


def top_mute(data, vp0, wind_length, offsets, dt, tdelay):

    taper = np.arange(wind_length)
    taper = np.sin(np.pi * taper / (2 * wind_length - 1)) ** 2
    nt = data.shape[0]

    for ii, off in enumerate(offsets):
        tmute = int((np.abs(off) / vp0 + 1.5 * tdelay) / dt)
        if tmute <= nt:
            data[0:tmute, ii] = 0
            mute_max = np.min([tmute + wind_length, nt])
            nmute = mute_max - tmute
            data[tmute:mute_max, ii] = data[tmute:mute_max, ii] * taper[:nmute]
        else:
            data[:, ii] = 0

    return data


def random_static(data, max_static):
    """
    Apply a random time shift (static) to each trace of data.
    Shifts can only be an integer of generator interval (for efficiency reason).

    @params:
    data  (numpy.array): The data array
    max_static   (float): Maximum number of samples that can be shifted

    @returns:
    (numpy.array): The data with random statics
    """
    ng = data.shape[1]
    shifts = (np.random.rand(ng) - 0.5) * max_static * 2
    for ii in range(ng):
        data[:, ii] = np.roll(data[:, ii], int(shifts[ii]), 0)
    return data


def random_noise(data, max_amp):
    """
    Add gaussian random noise to the data

    @params:
    data  (numpy.array): The data array
    max_amp   (float): Maximum amplitude of the noise relative to the data max

    @returns:
    (numpy.array): The data with noise
    """
    max_amp = max_amp * np.max(data) * 2.0
    data = data + (np.random.rand(data.shape[0], data.shape[1])-.5) * max_amp
    return data


def mute_nearoffset(data, max_off):
    """
    Randomly mute the near offset traces

    @params:
    data  (numpy.array): The data array
    max_amp   (float): Maximum offset that can be muted

    @returns:
    (numpy.array): The data with noise
    """
    data[:, :np.random.randint(max_off)] *= 0
    return data


def random_filt(data, filt_length):
    """
    Apply a random filter to the data

    @params:
    data  (numpy.array): The data array
    filt_length   (float):The filter length

    @returns:
    (numpy.array): The filtered data
    """
    filt_length = int((np.random.randint(filt_length) // 2) * 2 + 1)
    filt = np.random.rand(filt_length, 1)
    data = convolve2d(data, filt, 'same')
    return data


def random_time_scaling(data, dt, emin=-2.0, emax=2.0, scalmax=None):
    """
    Apply a random t**e gain to the data (dataout = data * t**e)

    @params:
    data  (numpy.array): The data array
    dt (float):          Time generator interval
    emin (float):        Minimum exponent of the t gain t**emin
    emax (float):        Maximum exponent of the t gain t**emax

    @returns:
    (numpy.array): The data with a random gain applied
    """
    t = np.reshape(np.arange(0, data.shape[0]) * dt, [data.shape[0], 1])
    e = np.random.rand() * (emax - emin) + emin
    scal = (t + 1e-6) ** e
    if scalmax is not None:
        scal[scal > scalmax] = scalmax
    return data * scal


def generate_reflections_ttime(vp, source_depth, dh, nt, dt, peak_freq,
                               tdelay, minoffset, identify_direct, tol=0.015,
                               window_width=0.2):
    """
    Generate an array with 1 at time of primary reflections for the minimum
    offset trace of a gather. Valid for a flat layered model.

    :param vp: A 1D array containing the Vp profile in depth
    :param source_depth: Depth of the source (in m)
    :param dh: Spatial grid size
    :param nt: Number of time steps
    :param dt: Sampling interval (in s)
    :param peak_freq: Peak frequency of the source
    :param tdelay: Delay of the source
    :param minoffset: Minimum offset of the gather
    :param identify_direct: Output an event of the direct arrival if True
    :param tol: The minimum relative velocity change to consider a reflection
    :param window_width: time window width in percentage of peak_freq

    :return: A 2D array with nt elements with 1 at reflecion times
             +- window_width/peak_freq, 0 elsewhere
    """

    vp = vp[int(source_depth / dh):]
    vlast = vp[0]
    ind = []
    for ii, v in enumerate(vp):
        if np.abs((v - vlast) / vlast) > tol:
            ind.append(ii - 1)
            vlast = v

    if minoffset != 0:
        delta = 2.0 * dh / vp
        t0 = np.cumsum(delta)
        vrms = np.sqrt(t0 * np.cumsum(vp**2 * delta))
        tref = np.sqrt(t0[ind]**2+minoffset**2/vrms[ind]**2) + tdelay
    else:
        ttime = 2 * np.cumsum(dh / vp) + tdelay
        tref = ttime[ind]

    if identify_direct:
        delta = 0
        if minoffset != 0:
            delta = minoffset / vp[0]
        tref = np.insert(tref, 0, tdelay + delta)

    tlabel = np.zeros(nt)
    for t in tref:
        imin = int(t / dt - window_width / peak_freq / dt)
        imax = int(t / dt + window_width / peak_freq / dt)
        if imin <= nt and imax <= nt:
            tlabel[imin:imax] = 1

    return tlabel


def two_way_travel_time(vp, dh, t0=0):
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
    t = 2 * np.cumsum(dh / vp) + t0

    return t


def vdepth2time(vp, dh, t, t0=0):
    """
    Converts interval velocity in depth to inverval velocity in time

    @params:
    vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
    pars (ModelParameter): Parameters used to generate the model

    @returns:
    vint (numpy.ndarray) : The interval velocity in time
    """
    tp = two_way_travel_time(vp, dh, t0)
    interpolator = interp1d(
        tp, vp, bounds_error=False, fill_value="extrapolate", kind="nearest",
    )
    vint = interpolator(t)

    return vint


# TODO Code interval velocity in time to interval velocity in depth.

# def vtime2depth(vint, t):
#     """
#     Converts interval velocity in time to interval velocity in depth
#
#     @params:
#     vp (numpy.ndarray) :  A 1D array containing the Vp profile in depth
#     pars (ModelParameter): Parameters used to generate the model
#
#     @returns:
#
#     vint (numpy.ndarray) : The interval velocity in time
#
#     """
#     #vpt, t = two_way_travel_time(vp, pars)
#     interpolator = interp1d(t*vint/2.0 + pars.source_depth, vint,
#                             bounds_error=False,
#                             fill_value="extrapolate",
#                             kind="linear")
#     vdepth = interpolator(np.arange(0, pars.NZ, 1) * pars.dh)
#
#     return vdepth


# TODO Recode calculate vrms into vint2vrms and simply it.
def vint2vrms(vint, t):
    dt = t[1:]-t[:-1]
    vrms = np.zeros_like(vint)
    vrms[:-1] = np.cumsum(dt * vint[:-1]**2)
    vrms[:-1] = np.sqrt(vrms[:-1] / (t[1:] - t[0]))
    vrms[-1] = vrms[-2]

    return vrms


def calculate_vrms(vp, dh, npad, nt, dt, tdelay, source_depth):
    """
    This method inputs vp and outputs the vrms. The global parameters in
    common.py are used for defining the depth spacing, source and receiver
    depth etc. This method assumes that source and receiver depths are same.

    The convention used is that the velocity denoted by the interval
    (i, i+1) grid points is given by the constant vp[i+1].

    @params:
    vp (numpy.ndarray) :  1D vp values in meters/sec.
    dh (float) : the spatial grid size
    npad (int) : Number of absorbing padding grid points over the source
    nt (int)   : Number of time steps of output
    dt (float) : Time step of the output
    tdelay (float): Time before source peak
    source_depth (float) The source depth in meters

    @returns:
    vrms (numpy.ndarray) : numpy array of shape (nt, ) with vrms
                           values in meters/sec.
    """
    nz = vp.shape[0]

    # Create a numpy array of depths corresponding to the vp grid locations.
    depth = np.arange(0, nz) * dh

    # Create a list of tuples of (relative depths, velocity) of the layers
    # following the depth of the source / receiver depths, till the last layer
    # before the padding zone at the bottom.
    last_depth = dh * (nz - npad - 1)
    rdepth_vel_pairs = [(d - source_depth, vp[i]) for i, d in enumerate(depth)
                        if source_depth < d <= last_depth]
    first_layer_vel = rdepth_vel_pairs[0][1]
    rdepth_vel_pairs.insert(0, (0.0, first_layer_vel))

    # Calculate a list of two-way travel times
    t = [
        2. * (rdepth_vel_pairs[index][0]-rdepth_vel_pairs[index - 1][0]) / vel
        for index, (_, vel) in enumerate(rdepth_vel_pairs) if index > 0
    ]
    t.insert(0, 0.0)
    total_time = 0.0
    for i, time in enumerate(t):
        total_time += time
        t[i] = total_time

    # The last time must be 'dt' * 'nt', so adjust the lists 'rdepth_vel_pairs'
    # and 't' by cropping and adjusting the last generator accordingly.
    rdepth_vel_pairs = [
        (rdepth_vel_pairs[i][0], rdepth_vel_pairs[i][1])
        for i, time in enumerate(t)
        if time <= nt * dt
    ]
    t = [time for time in t if time <= nt * dt]
    last_index = len(t) - 1
    extra_distance = (
        (nt*dt-t[last_index]) * rdepth_vel_pairs[last_index][1] / 2.
    )
    rdepth_vel_pairs[last_index] = (
        extra_distance + rdepth_vel_pairs[last_index][0],
        rdepth_vel_pairs[last_index][1]
    )
    t[last_index] = nt * dt

    # Compute vrms at the times in t.
    vrms = [first_layer_vel]
    sum_numerator = 0.0
    for i in range(1, len(t)):
        sum_numerator += (
            (t[i]-t[i - 1]) * rdepth_vel_pairs[i][1] * rdepth_vel_pairs[i][1]
        )
        vrms.append((sum_numerator / t[i])**.5)

    # Interpolate vrms to uniform time grid.
    tgrid = np.asarray(range(0, nt)) * dt
    vrms = np.interp(tgrid, t, vrms)
    vrms = np.reshape(vrms, [-1])
    # Adjust for time delay.
    t0 = int(tdelay / dt)
    vrms[t0:] = vrms[:-t0]
    vrms[:t0] = vrms[t0]

    return vrms


def smooth_velocity_wavelength(vp, dh, lt, lx):
    """
        Smooth a velocity model with a gaussian kernel proportional to the
        wavelength. Model is first transormed to interval velocity in time,
        then smoothed with a gaussian kernel with lt and lx standard deviation
        and retransformed to depth.

        @params:
        vp (numpy.ndarray) :  Velocity model
        dh (float) :  Grid spacing
        lt (float): Standard deviation of Gaussian kernel in time
        lx (float): Standard deviation of Gaussian kernel in x

        @returns:
        vp (numpy.ndarray):     Smoothed velocity model
    """
    vdepth = vp * 0
    dt = dh / np.max(vp) / 10.0
    nt = int(np.max(2 * np.cumsum(dh / vp, axis=0)) / dt)
    t = np.arange(1, nt+1, 1) * dt
    vint = np.zeros([nt, vp.shape[1]])
    for ii in range(0, vp.shape[1]):
        ti = 2 * np.cumsum(dh / vp[:, ii])
        ti = ti - ti[0]
        interpolator = interp1d(
            ti,
            vp[:, ii],
            bounds_error=False,
            fill_value="extrapolate",
            kind="nearest",
        )
        vint[:, ii] = interpolator(t)

    if lt > 0:
        vint = gaussian_filter(vint, [lt//dt, lx])

    for ii in range(0, vp.shape[1]):
        vavg = np.cumsum(dt*vint[:, ii]) / 2 / t
        vavg[0] = vint[0, ii]
        interpolator = interp1d(
            t * vavg,
            vint[:, ii],
            bounds_error=False,
            fill_value="extrapolate",
            kind="nearest",
        )
        vdepth[:, ii] = interpolator(np.arange(0, vp.shape[0], 1) * dh)

    return vdepth


def sortcmp(data, src_pos, rec_pos, binsize=None):
    """
        Sort data according to CMP positions

        @params:
        data (numpy.ndarray) :  Data Array nt X ntraces
        src_pos (numpy.ndarray) : SeisCL array containing source position
        rec_pos (numpy.ndarray): SeisCL array containing receiver position
        binsize (float): Bin size for CMPs

        @returns:
        data_cmp (numpy.ndarray) : Sorted data according to CMPs
        cmps (numpy.ndarray) :     X position of each cmp
    """
    if binsize is None:
        binsize = src_pos[0, 1] - src_pos[0, 0]

    sx = np.array([src_pos[0, int(srcid)] for srcid in rec_pos[3, :]])
    gx = rec_pos[0, :]
    cmps = ((sx + gx) / 2 / binsize).astype(int) * binsize
    offsets = sx - gx

    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    if data is not None:
        data_cmp = data[:, ind]
    else:
        data_cmp = None

    unique_cmps, counts = np.unique(cmps, return_counts=True)
    maxfold = np.max(counts)
    firstcmp = unique_cmps[np.argmax(counts == maxfold)]
    lastcmp = unique_cmps[-np.argmax(counts[::-1] == maxfold)-1]
    unique_cmps = unique_cmps[counts == maxfold]
    if data is not None:
        ind1 = np.argmax(cmps == firstcmp)
        ind2 = np.argmax(cmps > lastcmp)
        data_cmp = data_cmp[:, ind1:ind2]
        ncmps = unique_cmps.shape[0]
        data_cmp = np.reshape(data_cmp, [data_cmp.shape[0], maxfold, ncmps])

    return data_cmp, unique_cmps


def stack(cmp, times, offsets, velocities):
    """
        Compute the stacked trace of a list of CMP gathers

        @params:
        cmps (numpy.ndarray) :  CMP gathers nt X Noffset
        times (numpy.ndarray) : 1D array containing the time
        offsets (numpy.ndarray): 1D array containing the offset of each trace
        velocities (numpy.ndarray): 1D array nt containing the velocities

        @returns:
        stacked (numpy.ndarray) : a numpy array nt long containing the stacked
                                  traces of each CMP
        """

    return np.sum(nmo_correction(cmp, times, offsets, velocities), axis=1)


def semblance_gather(cmp, times, offsets, velocities):
    """
    Compute the semblance panel of a CMP gather

    @params:
    cmp (numpy.ndarray) :  CMP gather nt X Noffset
    times (numpy.ndarray) : 1D array containing the time
    offsets (numpy.ndarray): 1D array containing the offset of each trace
    velocities (numpy.ndarray): 1D array containing the test Nv velocities

    @returns:
    semb (numpy.ndarray) : numpy array ntxNv containing semblance
    """
    nt = cmp.shape[0]
    semb = np.zeros([nt, len(velocities)])
    for ii, vel in enumerate(velocities):
        nmo = nmo_correction(cmp, times, offsets, np.ones(nt) * vel)
        semb[:, ii] = semblance(nmo)

    return semb


def nmo_correction(cmp, times, offsets, velocities, stretch_mute=None):
    """
    Compute the NMO corrected CMP gather

    @params:
    cmp (numpy.ndarray) :  CMP gather nt X Noffset
    times (numpy.ndarray) : 1D array containing the time
    offsets (numpy.ndarray): 1D array containing the offset of each trace
    velocities (numpy.ndarray): 1D array containing the test nt velocities
                                in time

    @returns:
    nmo (numpy.ndarray) : array ntxNoffset containing the NMO corrected CMP
    """
    nmo = np.zeros_like(cmp)
    for j, x in enumerate(offsets):
        t = [
            reflection_time(t0, x, velocities[i])
            for i, t0 in enumerate(times)
        ]
        interpolator = CubicSpline(times, cmp[:, j], extrapolate=False)
        amps = np.nan_to_num(interpolator(t), copy=False)
        nmo[:, j] = amps
        if stretch_mute is not None:
            nmo[np.abs((times - t) / (times + 1e-10)) > stretch_mute, j] = 0
    return nmo


def reflection_time(t0, x, vnmo):
    """
    Compute the arrival time of a reflecion

    @params:
    t0 (float) :  Two-way travel-time in seconds
    x (float) :  Offset in meters
    vnmo (float): NMO velocity

    @returns:
    t (float): Reflection travel time
    """
    t = np.sqrt(t0 ** 2 + x ** 2 / vnmo ** 2)
    return t


def semblance(nmo_corrected, window=10):
    """
    Compute the semblance of a nmo corrected gather

    @params:
    nmo_corrected (numpy.ndarray) :  NMO corrected CMP gather nt X Noffset
    window (int): Number of time samples to average

    @returns:
    semblance (numpy.ndarray): Array ntx1 containing semblance
    """
    num = np.sum(nmo_corrected, axis=1) ** 2
    den = np.sum(nmo_corrected ** 2, axis=1) + 1e-12
    weights = np.ones(window) / window
    num = np.convolve(num, weights, mode='same')
    den = np.convolve(den, weights, mode='same')
    return num / den
