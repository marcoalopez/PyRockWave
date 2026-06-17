# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: ultrasonic.py                                                     #
# Description: This module contains various utilities for preprocessing       #
# and analysing pulse-echo ultrasonic signals.                                #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2026, Marco A. Lopez-Sanchez. All rights reserved.            #
#                                                                             #
# PyRockWave is free software: you can redistribute it and/or modify          #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# PyRockWave is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with PyRockWave. If not, see <http://www.gnu.org/licenses/>.          #
#                                                                             #
# Author: Marco A. Lopez-Sanchez                                              #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Email: lopezmarco [to be found at] uniovi dot es                            #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
# =========================================================================== #

# Import statements
import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.fft import fft, fftfreq


# Function definitions
def process_signal(
    raw_signal: npt.ArrayLike,
    roi_samples: tuple[int, int],
    sampling_rate_hz: float,
    auto_detrend: bool = True,
    apply_filter: dict | None = None,
) -> np.ndarray:
    """
    Pre-process pulse-echo ultrasound signal by cropping,
    detrending, and filtering.

    Parameters
    ----------
    raw_signal : array-like
        The raw ultrasound signal data.
    roi_samples : tuple of int
        A tuple (start, end) of **sample indices** (not times) defining
        the region of interest within the signal. Must satisfy
        ``0 <= start < end <= len(raw_signal)``. To select a region by
        time, convert seconds to samples first, e.g.
        ``int(round(t_seconds * sampling_rate_hz))``.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.
    auto_detrend : bool, optional
        If True, removes a linear trend from the cropped signal. If
        False, a constant detrend is still applied, i.e. the DC offset
        (mean) is removed; detrending is not skipped entirely.
        By default True.
    apply_filter : dict or None, optional, by default None
        If provided, should be a dictionary with keys:
        - 'lowcut': Lower frequency bound for the filter in Hz.
        - 'highcut': Upper frequency bound for the filter in Hz.
        - 'order': The order of the Butterworth filter (positive int).

    Returns
    -------
    np.ndarray
        The processed signal.

    Raises
    ------
    ValueError
        If the inputs are malformed or inconsistent (see
        :func:`_validate_process_signal`), or if the region of interest
        is too short for a zero-phase filter of the requested order.
    """

    # Ensure the signal is a numpy array (no copy if already an ndarray)
    signal = np.asarray(raw_signal)

    # Validate inputs
    _validate_process_signal(
        roi_samples, len(signal), sampling_rate_hz, apply_filter
    )

    # Crop signal to region of interest
    roi_signal = signal[roi_samples[0] : roi_samples[1]]

    # Apply detrending
    if auto_detrend:
        roi_signal = detrend(roi_signal, type="linear")
    else:
        roi_signal = detrend(roi_signal, type="constant")  # Only remove DC offset if not detrending linearly

    # Apply filter if specified
    if apply_filter is not None:
        # Calculate Nyquist frequency
        nyquist = 0.5 * sampling_rate_hz

        # Normalize filter frequencies
        low_normalized = apply_filter["lowcut"] / nyquist
        high_normalized = apply_filter["highcut"] / nyquist

        # Design zero-phase Butterworth bandpass filter
        sos = butter(
            N=apply_filter["order"],
            Wn=[low_normalized, high_normalized],
            btype="bandpass",
            analog=False,
            output="sos",
        )

        # sosfiltfilt pads the signal by 3 * (2 * n_sections + 1) samples and
        # requires the input to be longer than that padding. Check explicitly
        # so a short ROI raises a clear message instead of a cryptic SciPy one.
        padlen = 3 * (2 * sos.shape[0] + 1)
        if roi_signal.shape[0] <= padlen:
            raise ValueError(
                f"Region of interest ({roi_signal.shape[0]} samples) is too "
                f"short for a zero-phase Butterworth filter of order "
                f"{apply_filter['order']}; it must exceed {padlen} samples. "
                "Use a wider ROI or a lower filter order."
            )

        # Apply zero-phase filtering (forward and backward)
        # this results in zero phase distortion, which is important
        # for preserving the timing and shape of the ultrasound pulses.
        roi_signal = sosfiltfilt(sos, roi_signal)

    return roi_signal


def estimate_bandpass(
    signal: npt.ArrayLike,
    sampling_rate_hz: float,
    margin: float = 0.2,
) -> dict:
    """
    Estimate band-pass cut-off frequencies from a signal's amplitude
    spectrum using a half-maximum (-6 dB) threshold.

    UNTESTED!

    Parameters
    ----------
    signal : array-like
        The input signal, typically a cropped and detrended pulse.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.
    margin : float, optional
        Fractional padding added to each side of the detected band,
        expressed as a fraction of the band width, by default 0.2.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'lowcut' : float
            Lower cut-off frequency in Hz. May be negative for
            broadband or low-frequency signals; clip before use.
        - 'highcut' : float
            Upper cut-off frequency in Hz.
        - 'order' : int
            Suggested Butterworth filter order (fixed at 4).
        - 'peak_frequency' : float
            Frequency of the spectral peak in Hz.

    Notes
    -----
    The band is defined by the contiguous range of frequencies whose
    amplitude is at least 50% (-6 dB) of the spectral peak. The first
    and last frequencies above this threshold set the raw band edges,
    which are then widened by ``margin``. A single dominant band is
    assumed.

    Which to use
    ------------
    Prefer this estimator when you need the -6 dB bandwidth as an
    explicit spec, or when the spectrum has a sharply defined passband
    edge. Because the threshold is set by a single peak bin, it is more
    sensitive to noise and assumes one contiguous band. For typical,
    roughly symmetric narrowband ultrasonic pulses, the noise-robust
    :func:`estimate_bandpass_centroid` is usually the better default.
    The returned ``lowcut`` may be negative; clip it before use.
    """

    N = len(signal)
    dt = 1 / sampling_rate_hz

    # FFT
    yf = fft(signal)
    xf = fftfreq(N, dt)

    # Keep positive frequencies
    pos = xf > 0
    xf = xf[pos]
    spectrum = np.abs(yf[pos])

    peak_index = np.argmax(spectrum)
    f_peak = xf[peak_index]

    threshold = spectrum[peak_index] * 0.5
    band_indices = np.where(spectrum >= threshold)[0]

    lowcut = xf[band_indices[0]]
    highcut = xf[band_indices[-1]]

    # add a safety margin
    bandwidth = highcut - lowcut
    lowcut -= margin * bandwidth
    highcut += margin * bandwidth

    return {
        "lowcut": lowcut,
        "highcut": highcut,
        "order": 4,
        "peak_frequency": f_peak
    }


def estimate_bandpass_centroid(
    signal: npt.ArrayLike,
    sampling_rate_hz: float,
    sigma_mult: float = 2,
) -> dict:
    """
    Estimate band-pass cut-off frequencies from the spectral centroid
    and spectral spread of a signal.

    UNTESTED!

    Parameters
    ----------
    signal : array-like
        The input signal, typically a cropped and detrended pulse.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.
    sigma_mult : float, optional
        Number of spectral standard deviations on each side of the
        centroid used to set the band edges, by default 2.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'lowcut' : float
            Lower cut-off frequency in Hz. May be negative; clip
            before use.
        - 'highcut' : float
            Upper cut-off frequency in Hz.
        - 'order' : int
            Suggested Butterworth filter order (fixed at 4).
        - 'center_frequency' : float
            Amplitude-weighted spectral centroid in Hz.
        - 'bandwidth_sigma' : float
            Spectral standard deviation (spread) in Hz.

    Notes
    -----
    A Hann window is applied before the FFT to reduce spectral
    leakage. The centroid and spread are amplitude-weighted by the
    magnitude spectrum |X(f)|.

    Which to use
    ------------
    This is the recommended default for pulse-echo ultrasonic pulses:
    it uses the whole weighted spectrum, so it is robust to noise and a
    single spurious peak, and degrades gracefully. Use the -6 dB
    :func:`estimate_bandpass` instead when you need the explicit -6 dB
    bandwidth or the spectrum is asymmetric with a sharp passband edge.
    The returned ``lowcut`` may be negative; clip it before use.
    """

    N = len(signal)
    dt = 1 / sampling_rate_hz

    # Window to reduce spectral leakage
    window = np.hanning(N)
    signal = signal * window

    # FFT
    yf = fft(signal)
    xf = fftfreq(N, dt)

    # Keep positive frequencies
    mask = xf > 0
    xf = xf[mask]
    spectrum = np.abs(yf[mask])

    # Spectral centroid
    fc = np.sum(xf * spectrum) / np.sum(spectrum)

    # Spectral variance
    variance = np.sum(spectrum * (xf - fc)**2) / np.sum(spectrum)
    sigma = np.sqrt(variance)

    # Band limits
    lowcut = fc - sigma_mult * sigma
    highcut = fc + sigma_mult * sigma

    return {
        "lowcut": lowcut,
        "highcut": highcut,
        "order": 4,
        "center_frequency": fc,
        "bandwidth_sigma": sigma
    }


def trigger_sta_lta(
    signal: npt.ArrayLike,
    short_time_avg: int,
    long_time_avg: int,
) -> np.ndarray:
    """
    Computes the standard STA/LTA from a given signal.
    Adapted from obspy

    Parameters
    ----------
    signal : numpy.ndarray
        Seismic Trace
    short_time_avg : int
        Length of short time average window in samples
    long_time_avg : int
        Length of long time average window in samples

    Returns
    -------
    numpy.ndarray
        The STA/LTA characteristic function, the same length as
        ``signal``.
    """

    sta = np.cumsum(signal ** 2, dtype=np.float64)
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[short_time_avg:] = sta[short_time_avg:] - sta[:-short_time_avg]
    sta /= short_time_avg
    lta[long_time_avg:] = lta[long_time_avg:] - lta[:-long_time_avg]
    lta /= long_time_avg

    # Pad zeros
    sta[:long_time_avg - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


def _validate_process_signal(
    roi_samples: tuple[int, int],
    signal_length: int,
    sampling_rate_hz: float,
    apply_filter: dict | None,
) -> None:
    """
    Validate the inputs of :func:`process_signal`.

    Parameters
    ----------
    roi_samples : tuple of int
        A tuple (start, end) of sample indices defining the region of
        interest.
    signal_length : int
        Number of samples in the (un-cropped) signal, used to bound the
        region of interest.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.
    apply_filter : dict or None
        The band-pass filter specification, if any.

    Raises
    ------
    ValueError
        If any input is malformed or internally inconsistent.
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be a positive number.")

    if len(roi_samples) != 2:
        raise ValueError(
            "roi_samples must be a (start, end) tuple of length 2."
        )
    start, end = roi_samples
    if not (isinstance(start, (int, np.integer))
            and isinstance(end, (int, np.integer))):
        raise ValueError("roi_samples start and end must be integers.")
    if start < 0 or end <= start:
        raise ValueError("roi_samples must satisfy 0 <= start < end.")
    if end > signal_length:
        raise ValueError(
            f"roi_samples end ({end}) exceeds the signal length "
            f"({signal_length})."
        )

    if apply_filter is not None:
        required_keys = {"lowcut", "highcut", "order"}
        if not required_keys.issubset(apply_filter):
            raise ValueError(f"apply_filter must contain {required_keys}")

        order = apply_filter["order"]
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("apply_filter['order'] must be a positive integer.")

        nyquist = 0.5 * sampling_rate_hz
        lowcut = apply_filter["lowcut"]
        highcut = apply_filter["highcut"]
        if lowcut <= 0:
            raise ValueError("apply_filter['lowcut'] must be greater than 0.")
        if highcut <= lowcut:
            raise ValueError(
                "apply_filter['highcut'] must be greater than 'lowcut'."
            )
        if highcut >= nyquist:
            raise ValueError(
                "apply_filter['highcut'] must be below the Nyquist "
                f"frequency, i.e. <{nyquist}"
            )


# End of file
