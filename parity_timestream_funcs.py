import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.mixture as mix
from scipy.signal import periodogram as psd
from scipy.optimize import curve_fit
import os

def load_iq_csv(filename, time_col="time (ms)", iq_col="IQ"):
    """
    Load a fake SQUAT I/Q timestream from a CSV saved as:
        df = pd.DataFrame({"time (ms)": t, "IQ": iq})
        df.to_csv(filename, index=False)

    The IQ column is stored by pandas as complex-valued strings like
    "(1.23-0.45j)", so we parse them back to a complex numpy array.

    Returns
    -------
    times_ms : np.ndarray (float)
        Time array in milliseconds.
    iq : np.ndarray (complex)
        Complex I/Q timestream.
    """
    df = pd.read_csv(filename)
    times_ms = df[time_col].to_numpy(dtype=float)

    raw = df[iq_col].to_numpy()
    if np.iscomplexobj(raw):
        iq = raw.astype(complex)
    else:
        # Strip parentheses/whitespace, then parse each entry to complex.
        def _parse(s):
            return complex(str(s).strip().lstrip("(").rstrip(")").replace(" ", ""))
        iq = np.array([_parse(s) for s in raw], dtype=complex)

    return times_ms, iq

def compute_iq_rot(i, q):
    """
    Compute I and Q in the IQ_rot basis.

    Fits a 2-component Gaussian mixture to the raw IQ data, rotates so the
    two blob centers lie along the horizontal axis, and removes the vertical
    mean offset.

    Parameters
    ----------
    i, q : np.ndarray
        Raw I and Q data vectors.

    Returns
    -------
    i_rot, q_rot : np.ndarray
        Rotated I and Q data.
    raw_gmm : sklearn.mixture.GaussianMixture
        The 2-component GMM fit to the raw IQ data (needed by compute_iq_phase).
    """
    # - fit 2-component gaussian mixture to the 2d IQ data
    iq_vect = np.zeros((i.shape[0], 2))
    iq_vect[:, 0] = i
    iq_vect[:, 1] = q
    raw_gmm = mix.GaussianMixture(n_components=2)
    raw_gmm.fit(iq_vect)

    imeans, qmeans = raw_gmm.means_.transpose()
    i0_center, i1_center = imeans
    q0_center, q1_center = qmeans
    rot_i = i1_center - i0_center
    rot_q = q1_center - q0_center

    # - determine the rotation angle, account for the quadrant
    rot_angle = -1 * np.arctan2(rot_q, rot_i)

    # - rotate into the IQ_rot basis, remove the mean offset in the vertical axis
    i_rot, q_rot = rotate_iq(i, q, rot_angle)
    q_rot -= q_rot.mean()

    # - fit a temporary gmm, translate horizontally to center the blobs
    gmm_tmp = mix.GaussianMixture(n_components=2)
    iq_vect = np.zeros((i_rot.shape[0], 2))
    iq_vect[:, 0] = i_rot
    iq_vect[:, 1] = q_rot
    gmm_tmp.fit(iq_vect)
    imeans, qmeans = gmm_tmp.means_.transpose()
    i_rot -= imeans.mean()

    return i_rot, q_rot, raw_gmm


def rotate_iq(i, q, angle):
    """Rotate I/Q data by `angle` radians (static helper from the class)."""
    Irot = i * np.cos(angle) - q * np.sin(angle)
    Qrot = i * np.sin(angle) + q * np.cos(angle)
    return Irot, Qrot

def calc_psd(data, times_ms, nfft=2 ** 12):
    """
    Calculate the power spectral density of a 1D timestream.

    Normalizes the data in the time domain using the telegraph centerline
    (the mean) and the separation between the two telegraph-state means,
    then computes the periodogram.

    Parameters
    ----------
    data : np.ndarray (real)
        1D data to take the PSD of (e.g. Phase_phase, or iq.real).
    times_ms : np.ndarray
        Time array in milliseconds.
    nfft : int
        Number of points to use in the FFT.

    Returns
    -------
    psd_freqs : np.ndarray
        Frequency axis (Hz).
    psd_yvals : np.ndarray
        PSD values.
    tel_cen : float
        Telegraph centerline used for normalization.
    tel_sep : float
        Telegraph state separation used for normalization.
    """
    # - threshold is simply the mean of the data (from _calc_threshold)
    centerline = np.mean(data)
    tel_state1 = np.mean(data[data < centerline])
    tel_state2 = np.mean(data[data > centerline])
    tel_separation = np.abs(tel_state2 - tel_state1)
    datavals = (data - centerline) / tel_separation

    # - compute the periodogram
    times_sec = times_ms / 1e3
    dt = times_sec[1] - times_sec[0]
    f, p = psd(datavals, fs=1 / dt, nfft=nfft)

    return f, p, centerline, tel_separation

def lorentzian_psd(f, S0, fc):
    """Standard Lorentzian PSD."""
    return S0 / (1 + (f / fc) ** 2)


def corrected_lorentzian_psd(f, S0, F, Gamma_p, delta_t):
    """Lorentzian PSD with noise-floor correction (eq.14 of arXiv:2402.15471)."""
    return S0 * ((4 * F ** 2 * Gamma_p) / ((2 * Gamma_p) ** 2 + (2 * np.pi * f) ** 2)
                 + (1 - F ** 2) * delta_t)


def fit_psd(psd_freqs, psd_yvals, times_ms, n_files=1,
            fit_to_corrected_function=True, min_freq_bound=None):
    """
    Fit a Lorentzian to PSD data.

    Parameters
    ----------
    psd_freqs, psd_yvals : np.ndarray
        Frequency axis and PSD values (from calc_psd).
    times_ms : np.ndarray
        Time array in ms (used to compute delta_t for the corrected fit).
    n_files : int
        Number of averaged files, used to weight the fit (default 1).
    fit_to_corrected_function : bool
        If True, fit the noise-floor-corrected Lorentzian; else the standard one.
    min_freq_bound : float or None
        Minimum frequency bound [Hz] for the fit. If None, no bound.

    Returns
    -------
    fit_params : dict or None
        Dictionary of fit parameters, or None if the fit fails.
    """
    try:
        S0_guess = np.max(psd_yvals)
        peak_freq = psd_freqs[np.argmax(psd_yvals)]
        freq_bound = 0 if min_freq_bound is None else min_freq_bound

        if fit_to_corrected_function:
            F_guess = 0.9
            delta_t = (times_ms[1] - times_ms[0]) / 1e3
            popt, pcov = curve_fit(
                lambda f, S0, F, Gamma_p: corrected_lorentzian_psd(f, S0, F, Gamma_p, delta_t),
                psd_freqs[1:], psd_yvals[1:],
                p0=[S0_guess, F_guess, peak_freq],
                sigma=abs(psd_yvals[1:]) / np.sqrt(n_files),
                absolute_sigma=True,
                bounds=([-np.inf, 0, freq_bound], [np.inf, 1, np.inf]),
            )
            fit_params = {
                'psd_corr_fit_S0': popt[0],
                'psd_corr_fit_F': popt[1],
                'psd_corr_fit_Gamma_p': popt[2],
                'psd_corr_fit_delta_t': delta_t,
                'psd_corr_fit_pcov': pcov,
            }
        else:
            popt, pcov = curve_fit(
                lorentzian_psd,
                psd_freqs[1:], psd_yvals[1:],
                p0=[S0_guess, peak_freq],
                sigma=abs(psd_yvals[1:]) / np.sqrt(n_files),
                absolute_sigma=True,
                bounds=([-np.inf, freq_bound], [np.inf, np.inf]),
            )

            fit_params = {
                'psd_fit_S0': popt[0],
                'psd_fit_fc': popt[1],
                'psd_fit_pcov': pcov,
            }

        return fit_params

    except Exception as e:
        print(f"Error during fitting: {e}")
        return None

def plot_psd(psd_freqs, psd_yvals, plot_title=None,
             ymin=1e-12, ymax=1, savepath=None, ax=None):
    """
    Plot PSD data on log-log axes.

    Parameters
    ----------
    psd_freqs, psd_yvals : np.ndarray
        Frequency axis (Hz) and PSD values, from calc_psd().
    plot_title : str or None
        Title for the plot.
    ymin, ymax : float or None
        Y-axis limits. If either is None, autoscaling is used.
    savepath : str or None
        If given, the figure is saved to this path.
    ax : matplotlib Axes or None
        If provided, plots onto it rather than creating a new figure.

    Returns
    -------
    fig, ax
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel='Frequency (Hz)', ylabel='Power spectral density',
           title=plot_title)

    ax.loglog(psd_freqs, psd_yvals)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)

    return fig, ax

def plot_psd_fit(psd_freqs, psd_yvals, fit_params, times_ms,
                 plot_title=None, ymin=None, ymax=None,
                 plot_corrected_fit=True, plot_standard_fit=True,
                 linear_ax=False, savepath=None, ax=None):
    """
    Plot PSD data along with the Lorentzian fit(s).

    Parameters
    ----------
    psd_freqs, psd_yvals : np.ndarray
        Frequency axis (Hz) and PSD values, from calc_psd().
    fit_params : dict
        Fit-parameter dict returned by fit_psd(). May contain standard-fit
        keys ('psd_fit_S0', 'psd_fit_fc') and/or corrected-fit keys
        ('psd_corr_fit_S0', 'psd_corr_fit_F', 'psd_corr_fit_Gamma_p', ...).
    times_ms : np.ndarray
        Time array (ms), used to compute delta_t for the corrected fit curve.
    plot_title : str or None
        Title for the plot.
    ymin, ymax : float or None
        Y-axis limits. If None, guessed from the data.
    plot_corrected_fit, plot_standard_fit : bool
        Which fit curve(s) to overlay (only plotted if present in fit_params).
    linear_ax : bool
        If True, use a linear y-axis (semilogx); else log-log.
    savepath : str or None
        If given, the figure is saved to this path.
    ax : matplotlib Axes or None
        If provided, plots onto it rather than creating a new figure.

    Returns
    -------
    fig, ax
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel='Frequency [Hz]', ylabel='Power spectral density',
           title=plot_title)

    # - Decide which fits are actually available in fit_params
    if fit_params is None:
        fit_params = {}
    has_standard = 'psd_fit_S0' in fit_params and fit_params.get('psd_fit_S0') is not None
    has_corrected = 'psd_corr_fit_S0' in fit_params and fit_params.get('psd_corr_fit_S0') is not None
    plot_standard_fit = plot_standard_fit and has_standard
    plot_corrected_fit = plot_corrected_fit and has_corrected

    data_color = 'steelblue'
    fit_color = 'indigo'
    corrected_fit_color = 'indianred'

    delta_t = (times_ms[1] - times_ms[0]) / 1e3
    plot_fn = ax.semilogx if linear_ax else ax.loglog

    # - data
    plot_fn(psd_freqs, psd_yvals, label='Data', color=data_color)

    # - standard Lorentzian fit
    if plot_standard_fit:
        plot_fn(psd_freqs,
                lorentzian_psd(psd_freqs, fit_params['psd_fit_S0'],
                               fit_params['psd_fit_fc']),
                label=f'Fc = {fit_params["psd_fit_fc"]:.2f} Hz',
                color=fit_color, linewidth=3)

    # - corrected Lorentzian fit
    if plot_corrected_fit:
        gamma_err = np.sqrt(np.diag(fit_params['psd_corr_fit_pcov'])[2])
        plot_fn(psd_freqs,
                corrected_lorentzian_psd(psd_freqs,
                                         fit_params['psd_corr_fit_S0'],
                                         fit_params['psd_corr_fit_F'],
                                         fit_params['psd_corr_fit_Gamma_p'],
                                         delta_t),
                label=f'Γ_p = {fit_params["psd_corr_fit_Gamma_p"]:.2f} '
                      f'+- {gamma_err:.2f} Hz',
                color=corrected_fit_color, linewidth=3)

    ax.grid()

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    else:
        # - guess limits from the data
        ax.set_ylim(np.min(psd_yvals[2:-2]) * 0.1,
                    np.max(psd_yvals[2:-2]) * 50)

    ax.legend()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)

    return fig, ax