import numpy as np
import warnings

def _check_nyquist(switching_rate, sampling_rate, warn_fraction = 0.1):
    nyquist = sampling_rate / 2.0
    if switching_rate >= nyquist:
        raise ValueError(
            f"switching_rate ({switching_rate} Hz) exceeds the Nyquist "
            f"frequency ({nyquist} Hz). Increase sampling_rate."
        )

    if warn_fraction and switching_rate > warn_fraction * nyquist:
        warnings.warn(
            f"switching_rate ({switching_rate} Hz) is a large fraction of "
            f"Nyquist ({nyquist} Hz); transitions will be poorly resolved.",
            RuntimeWarning,
        )

    return None

def generate_telegraph_signal(
    duration,
    sampling_rate,
    switching_rate,
    amplitude=1.0,
    offset=0.0,
    initial_state=None,
    rng=None,
):
    """
    Generate a two-level random telegraph signal (RTS/RTN).

    The signal switches between two levels via a Poisson process, so the
    dwell time in each state is exponentially distributed with mean
    1 / switching_rate.

    Parameters
    ----------
    duration : float
        Total length of the timestream in seconds.
    sampling_rate : float
        Samples per second (Hz).
    switching_rate : float
        Mean switching rate of the telegraph signal (Hz).
    amplitude : float, optional
        Peak-to-peak amplitude of the telegraph signal. The two levels are
        placed at offset +/- amplitude/2. (Amplitude is physically irrelevant and usually in ADC units)
    offset : float, optional
        DC offset added to both levels.
    initial_state : int or None, optional
        Starting state, 0 or 1. If None, chosen at random.
    rng : numpy.random.Generator or None, optional
        Random generator for reproducibility. If None, a fresh one is made.

    Returns
    -------
    t : np.ndarray
        Time array in seconds.
    signal : np.ndarray
        The (noiseless) telegraph signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = 1.0 / sampling_rate
    n = int(round(duration * sampling_rate))
    _check_nyquist(switching_rate, sampling_rate)          # <-- caller owns this
    state = _generate_telegraph_state(n, dt, switching_rate,
                                      initial_state, rng)
    t = np.arange(n) * dt
    signal = offset + (state - 0.5) * amplitude
    return t, signal


def add_gaussian_noise(signal, noise_sigma=None, snr=None, amplitude=1.0, rng=None):
    """
    Add white Gaussian noise to a signal.

    You can specify the noise either directly (noise_sigma) or via a
    signal-to-noise ratio (snr) relative to the telegraph amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    noise_sigma : float or None
        Standard deviation of the Gaussian noise. Takes priority if given.
    snr : float or None
        Signal-to-noise ratio defined as amplitude / noise_sigma.
        Used only if noise_sigma is None.
    amplitude : float
        Telegraph amplitude, used with `snr` to compute noise_sigma.
    rng : numpy.random.Generator or None

    Returns
    -------
    np.ndarray
        Signal with added noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    if noise_sigma is None:
        if snr is None:
            raise ValueError("Provide either noise_sigma or snr.")
        noise_sigma = amplitude / snr

    return signal + rng.normal(0.0, noise_sigma, size=signal.shape)


def generate_squat_data(
    duration,
    sampling_rate,
    switching_rate,
    amplitude=1.0,
    snr=5.0,
    noise_sigma=None,
    offset=0.0,
    seed=None,
):
    """
    Generate a full fake SQUAT timestream: telegraph signal + Gaussian noise.

    Parameters
    ----------
    duration : float
        Length of timestream in seconds.
    sampling_rate : float
        Sampling rate in Hz
    switching_rate : float
        Telegraph switching rate in Hz
    amplitude : float, optional
        Telegraph peak-to-peak amplitude (default 1).
    snr : float, optional
        Signal-to-noise ratio (amplitude / noise_sigma). Ignored if
        noise_sigma is given.
    noise_sigma : float or None, optional
        Explicit Gaussian noise std. Overrides snr if provided.
    offset : float, optional
        DC offset.
    seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time array in seconds.
    data : np.ndarray
        Noisy timestream.
    clean : np.ndarray
        The underlying noiseless telegraph signal.
    """
    rng = np.random.default_rng(seed)

    t, clean = generate_telegraph_signal(
        duration=duration,
        sampling_rate=sampling_rate,
        switching_rate=switching_rate,
        amplitude=amplitude,
        offset=offset,
        rng=rng,
    )

    data = add_gaussian_noise(
        clean, noise_sigma=noise_sigma, snr=snr, amplitude=amplitude, rng=rng
    )

    return t, data, clean

###########################
########## I Q ############
###########################

def _generate_telegraph_state(n_samples, dt, rate, initial_state=None, rng=None):
    """
    Generate the underlying two-level state sequence (0/1) of a Poissonian
    random telegraph signal. This is the shared core used by both the scalar
    and complex (I/Q) generators.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    dt : float
        Time step between samples (s).
    rate : float or np.ndarray
        Switching rate (Hz). Scalar for a constant rate, or an array of
        length n_samples for a time-varying rate.
    initial_state : int or None
        Starting state 0/1; random if None.
    rng : np.random.Generator or None

    Returns
    -------
    state : np.ndarray
        Integer 0/1 state sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Exact two-level flip probability (odd number of Poisson switches).
    # Broadcasts whether `rate` is scalar or array.
    p_flip = 0.5 * (1.0 - np.exp(-2.0 * np.asarray(rate) * dt))

    flips = rng.random(n_samples) < p_flip
    if initial_state is None:
        initial_state = rng.integers(0, 2)
    flips[0] = bool(initial_state)

    return np.mod(np.cumsum(flips), 2)


def generate_squat_iq_data(
    duration,
    sampling_rate,
    switching_rate,
    state0_iq=(1.0, 0.0),
    state1_iq=None,
    separation=None,
    angle=0.0,
    center=(0.0, 0.0),
    dc_offset=(0.0, 0.0),
    snr=5.0,
    noise_sigma=None,
    initial_state=None,
    seed=None,
):
    """
    Generate fake complex (I/Q) SQUAT timestream data.

    Each of the two telegraph states corresponds to a fixed point in the IQ
    plane (as in dispersive transmon readout). The signal jumps between these
    two points via a Poisson process, and independent white Gaussian noise is
    added to each quadrature.

    Parameters
    ----------
    duration : float
        Length of timestream in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    switching_rate : float
        Telegraph switching rate in Hz.
    state0_iq, state1_iq : tuple of float, optional
        (I, Q) coordinates of the two telegraph levels. You can either give
        both explicitly, or give state0_iq plus `separation`/`angle` to place
        state1 automatically (see below). If both are None, a default
        symmetric pair is used.
    separation : float or None, optional
        If state1_iq is None, place the two states this far apart in the IQ
        plane, along direction `angle`, symmetric about `center`. This is a
        convenient single knob for "blob separation".
    angle : float, optional
        Angle (radians) of the separation axis when using `separation`.
    center : tuple of float, optional
        Midpoint of the two states in the IQ plane when using `separation`.
    dc_offset : tuple of float or complex, optional
        A constant (I, Q) offset added to both states, so the
        pair of IQ blobs is shifted away from the origin as a whole.
        Accepts either a 2-tuple (I, Q) or a complex number I + 1j*Q.
        This models mixer leakage / cable-delay offsets seen in real data.
        Note: `center` sets the midpoint *before* this global offset; the
        two combine additively, so you can use either or both.
    snr : float, optional
        Signal-to-noise ratio, defined as (state separation) / noise_sigma.
        The same noise_sigma is applied to both I and Q. Ignored if
        noise_sigma is given.
    noise_sigma : float or None, optional
        Explicit per-quadrature Gaussian noise std. Overrides snr.
    initial_state : int or None, optional
        Starting telegraph state (0 or 1). Random if None.
    seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time array in seconds.
    iq : np.ndarray (complex)
        Noisy complex timestream, iq = I + 1j*Q.
    clean : np.ndarray (complex)
        Underlying noiseless complex telegraph signal.
    """
    rng = np.random.default_rng(seed)

    # --- Resolve the two IQ points ------------------------------------
    p0 = np.asarray(state0_iq, dtype=float)

    if state1_iq is not None:
        p1 = np.asarray(state1_iq, dtype=float)
    elif separation is not None:
        # Place the two states symmetrically about `center` along `angle`.
        c = np.asarray(center, dtype=float)
        direction = np.array([np.cos(angle), np.sin(angle)])
        p0 = c - 0.5 * separation * direction
        p1 = c + 0.5 * separation * direction
    else:
        # Default: mirror state0 through the origin for a symmetric pair.
        p1 = -p0

    z0 = p0[0] + 1j * p0[1]
    z1 = p1[0] + 1j * p1[1]
    sep = np.abs(z1 - z0)

    # --- Apply a DC offset to both states --------------------
    # Accept either a (I, Q) tuple or a complex number.
    dc = np.asarray(dc_offset)
    if dc.dtype == complex or dc.ndim == 0:
        z_offset = complex(dc_offset)
    else:
        z_offset = dc_offset[0] + 1j * dc_offset[1]
    z0 += z_offset
    z1 += z_offset

    # --- Build the state sequence and map to IQ points ----------------
    dt = 1.0 / sampling_rate
    n_samples = int(round(duration * sampling_rate))
    _check_nyquist(switching_rate, sampling_rate)

    state = _generate_telegraph_state(
        n_samples, dt, switching_rate,
        initial_state=initial_state, rng=rng,
    )
    t = np.arange(n_samples) * dt

    clean = np.where(state == 0, z0, z1)

    # --- Add independent Gaussian noise to I and Q --------------------
    if noise_sigma is None:
        if sep == 0:
            raise ValueError(
                "State separation is zero; provide noise_sigma explicitly "
                "or give two distinct IQ points."
            )
        noise_sigma = sep / snr

    noise = rng.normal(0.0, noise_sigma, size=clean.shape) \
        + 1j * rng.normal(0.0, noise_sigma, size=clean.shape)
    iq = clean + noise

    return t, iq, clean



############### fake event generator ################

def _event_switching_rate(t, baseline_rate, event_time, peak_rate, tau):
    """
    Time-varying switching rate: a baseline with an exponentially-decaying
    spike starting at `event_time`.

        lambda(t) = baseline                                    for t <  event_time
        lambda(t) = baseline + peak * exp(-(t-event_time)/tau)  for t >= event_time
    """
    rate = np.full_like(t, baseline_rate, dtype=float)
    after = t >= event_time
    rate[after] += peak_rate * np.exp(-(t[after] - event_time) / tau)
    return rate


def generate_squat_iq_data_with_event(
    duration,
    sampling_rate,
    baseline_switching_rate,
    event_time,
    event_peak_rate=100e3,
    event_tau=2.5e-3,
    state0_iq=(1.0, 0.0),
    state1_iq=None,
    separation=None,
    angle=0.0,
    center=(0.0, 0.0),
    dc_offset=(0.0, 0.0),
    snr=5.0,
    noise_sigma=None,
    initial_state=None,
    oversample=None,
    seed=None,
):
    """
    Generate fake complex (I/Q) SQUAT timestream data that includes a
    particle-impact event.

    The event manifests as a sudden spike in the telegraph switching rate at
    `event_time`, peaking at `event_peak_rate` and decaying exponentially with
    time constant `event_tau`. When the instantaneous switching rate exceeds
    the sampling rate, many switches occur within one sample interval, so the
    recorded I/Q value is the time-average of the two states -> the two blobs
    "blur" together toward their midpoint. This is reproduced by simulating on
    a fine oversampled grid and averaging down to `sampling_rate`.

    Parameters
    ----------
    duration : float
        Length of timestream in seconds.
    sampling_rate : float
        Output sampling rate in Hz (intended range 1e3 - 1e5).
    baseline_switching_rate : float
        Quiescent telegraph switching rate away from the event (Hz).
    event_time : float
        Time of the particle impact (s).
    event_peak_rate : float, optional
        Peak additional switching rate at impact (Hz). Default 100 kHz.
    event_tau : float, optional
        Exponential decay constant of the event (s). Default 2.5 ms
        (=> the event is largely gone after ~4*tau ~ 10 ms).
    state0_iq, state1_iq, separation, angle, center, dc_offset :
        IQ-placement parameters, identical to generate_squat_iq_data.
    snr, noise_sigma :
        Noise specification, identical to generate_squat_iq_data. Noise is
        added AFTER the fine-grid averaging, at the output sampling rate.
    initial_state : int or None, optional
        Starting telegraph state (0 or 1). Random if None.
    oversample : int or None, optional
        Fine-grid oversampling factor relative to sampling_rate. If None, it
        is chosen automatically so the fine grid resolves the event peak rate
        with margin (>= ~10x event_peak_rate).
    seed : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Output time array in seconds (at sampling_rate).
    iq : np.ndarray (complex)
        Noisy complex timestream, iq = I + 1j*Q.
    clean : np.ndarray (complex)
        Underlying noiseless complex signal AFTER blur-averaging (so during
        the event it takes intermediate values between the two blobs).
    """
    rng = np.random.default_rng(seed)

    # Resolve the two IQ points
    p0 = np.asarray(state0_iq, dtype=float)
    if state1_iq is not None:
        p1 = np.asarray(state1_iq, dtype=float)
    elif separation is not None:
        c = np.asarray(center, dtype=float)
        direction = np.array([np.cos(angle), np.sin(angle)])
        p0 = c - 0.5 * separation * direction
        p1 = c + 0.5 * separation * direction
    else:
        p1 = -p0

    z0 = p0[0] + 1j * p0[1]
    z1 = p1[0] + 1j * p1[1]
    sep = np.abs(z1 - z0)

    # Global DC offset (accepts (I, Q) tuple or complex).
    dc = np.asarray(dc_offset)
    if dc.dtype == complex or dc.ndim == 0:
        z_offset = complex(dc_offset)
    else:
        z_offset = dc_offset[0] + 1j * dc_offset[1]
    z0 += z_offset
    z1 += z_offset

    # Choose the fine-grid oversampling factor
    # We need the fine grid to resolve the peak event rate. Aim for at least
    # ~10 fine samples per mean dwell time at the peak rate.
    if oversample is None:
        needed = 10.0 * event_peak_rate / sampling_rate
        oversample = max(1, int(np.ceil(needed)))

    fine_rate = sampling_rate * oversample
    n_out = int(round(duration * sampling_rate))
    n_fine = n_out * oversample

    nyquist = 2.0 * event_peak_rate
    if fine_rate < nyquist:
        warnings.warn(
            f"Fine grid ({fine_rate:.3g} Hz) under-resolves the event peak "
            f"rate ({event_peak_rate:.3g} Hz); increase `oversample`.",
            RuntimeWarning,
        )

    t_fine = np.arange(n_fine) / fine_rate

    # --- Time-varying switching rate and telegraph on the fine grid ---
    rate_fine = _event_switching_rate(
        t_fine, baseline_switching_rate, event_time, event_peak_rate, event_tau
    )
    state_fine = _generate_telegraph_state(
        n_fine, 1.0 / fine_rate, rate_fine,   # rate_fine is an array
        initial_state=initial_state, rng=rng,
    )
    clean_fine = np.where(state_fine == 0, z0, z1)

    # --- Average the fine grid down to the output sampling rate -------
    # This block-average over each output sample is what produces the "blur":
    # when many switches fall inside one sample, the mean lands between blobs.
    clean = clean_fine.reshape(n_out, oversample).mean(axis=1)
    t = np.arange(n_out) / sampling_rate

    # --- Add noise at the output rate ---------------------------------
    if noise_sigma is None:
        if sep == 0:
            raise ValueError(
                "State separation is zero; provide noise_sigma explicitly "
                "or give two distinct IQ points."
            )
        noise_sigma = sep / snr

    noise = rng.normal(0.0, noise_sigma, size=clean.shape) \
        + 1j * rng.normal(0.0, noise_sigma, size=clean.shape)
    iq = clean + noise

    return t, iq, clean