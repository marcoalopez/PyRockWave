# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: ultrasonic.py                                                     #
# Description: This module contains various utilities for preprocessing       #
# and analysing pulse-echo ultrasonic signals.                                                            #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2023-present, Marco A. Lopez-Sanchez. All rights reserved.    #
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
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.fft import fft, fftfreq


# Function definitions
def process_signal(
    raw_signal,
    roi,
    sampling_rate_hz,
    auto_detrend=True,
    apply_filter=None
):
    """
    Pre-process pulse-echo ultrasound signal by cropping,
    detrending, and filtering.

    Parameters
    ----------
    raw_signal : array-like
        The raw ultrasound signal data.
    roi : tuple
        A tuple (start, end) defining the region of
        interest within the signal.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.
    auto_detrend : bool, optional
        If True, removes the linear trend from the signal,
        by default True.
    apply_filter : dict or None, optional, by default None
        If provided, should be a dictionary with keys:
        - 'lowcut': Lower frequency bound for the filter in Hz.
        - 'highcut': Upper frequency bound for the filter in Hz.
        - 'order': The order of the filter.

    Returns
    -------
    np.ndarray
        The processed signal.
    """

    # Ensure the signal is a numpy array
    signal = np.array(raw_signal)

    # Crop signal to region of interest
    roi_signal = signal[roi[0] : roi[1]]

    # Apply detrending
    if auto_detrend:
        roi_signal = detrend(roi_signal, type="linear")
    else:
        roi_signal = detrend(roi_signal, type="constant")  # Only remove DC offset if not detrending linearly

    # Apply filter if specified
    if apply_filter:
        required_keys = {"lowcut", "highcut", "order"}
        if not required_keys.issubset(apply_filter):
            raise ValueError(f"apply_filter must contain {required_keys}")

        # Calculate Nyquist frequency
        nyquist = 0.5 * sampling_rate_hz
        if apply_filter["highcut"] >= nyquist:
            raise ValueError(f"Selected filter highcut must be below Nyquist, i.e. <{nyquist/2}")

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

        # Apply zero-phase filtering (forward and backward)
        # this results in zero phase distortion, which is important
        # for preserving the timing and shape of the ultrasound pulses.
        roi_signal = sosfiltfilt(sos, roi_signal)

    return roi_signal


def estimate_bandpass(
    signal,
    sampling_rate_hz,
    margin=0.2
) -> dict:
    """
    _summary_

    UNTESTED!

    Parameters
    ----------
    signal : _type_
        _description_
    sampling_rate_hz : _type_
        _description_
    margin : float, optional
        _description_, by default 0.2

    Returns
    -------
    dict
        _description_
    """

    N = len(signal)
    dt = 1 / sampling_rate_hz

    yf = fft(signal)
    xf = fftfreq(N, dt)

    pos = xf > 0
    xf = xf[pos]
    spectrum = np.abs(yf[pos])

    peak_index = np.argmax(spectrum)
    f_peak = xf[peak_index]

    threshold = spectrum[peak_index] * 0.5
    band_indices = np.where(spectrum >= threshold)[0]

    lowcut = xf[band_indices[0]]
    highcut = xf[band_indices[-1]]

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
    signal,
    sampling_rate_hz,
    sigma_mult=2
) -> dict:
    """
    _summary_

    UNTESTED!

    Parameters
    ----------
    signal : _type_
        _description_
    sampling_rate_hz : _type_
        _description_
    sigma_mult : int, optional
        _description_, by default 2

    Returns
    -------
    dict
        _description_
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
    signal,
    short_time_avg,
    long_time_avg
) -> np.array:
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
    _type_
        _description_
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


# End of file
