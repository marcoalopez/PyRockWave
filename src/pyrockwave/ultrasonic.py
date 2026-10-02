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
import warnings

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt, detrend, hilbert, find_peaks
from scipy.signal.windows import tukey
from scipy.fft import rfft, rfftfreq


# Function definitions
def process_signal(
    raw_signal: npt.ArrayLike,
    roi_samples: tuple[int, int],
    sampling_rate_hz: float,
    auto_detrend: bool = True,
    apply_filter: bool = False,
    return_filter_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict | None]:
    """
    Pre-process pulse-echo ultrasound signal by cropping,
    detrending, and (optionally) band-pass filtering.

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
        The digitizer sampling rate in Hz, i.e. ``1 / dt`` where ``dt``
        is the time step between consecutive samples (e.g.
        ``1 / (t[1] - t[0])`` for a time column ``t`` in seconds).
        This is a property of the acquisition, not of the transducer:
        do not pass the transducer's nominal centre frequency.
    auto_detrend : bool, optional
        If True, removes a linear trend from the cropped signal. If
        False, a constant detrend is still applied, i.e. the DC offset
        (mean) is removed; detrending is not skipped entirely.
        By default True.
    apply_filter : bool, optional
        If True, estimate the pass band with :func:`estimate_bandpass`
        (-6 dB band plus margin) from the strongest echo of the cropped
        and detrended signal, isolated with :func:`isolate_echoes` and
        :func:`gate_echo` (default settings), and apply a zero-phase
        Butterworth band-pass filter to the whole region of interest.
        If False, the signal is left unfiltered. By default False.
    return_filter_params : bool, optional
        If True, also return the band-pass parameters used for
        filtering (``None`` when ``apply_filter`` is False).
        By default False.

    Returns
    -------
    np.ndarray or tuple of (np.ndarray, dict or None)
        The processed signal. If ``return_filter_params`` is True, a
        tuple ``(signal, filter_params)`` where ``filter_params`` is the
        dictionary returned by :func:`estimate_bandpass`, or ``None`` if
        no filter was applied.

    Raises
    ------
    ValueError
        If the inputs are malformed or inconsistent (see
        :func:`_validate_process_signal`), or if the region of interest
        is too short for a zero-phase filter of the requested order.

    Notes
    -----
    The pass band is estimated from a single gated echo, as in the
    -6 dB bandwidth measurement of ASTM E1065, rather than from the
    whole region of interest. Two echoes separated by a delay ``T``
    produce interference ripples in the spectrum with a period of
    ``1 / T``; when the ripple dips fall below the -6 dB threshold
    they truncate the estimated band, which can collapse to the ripple
    period and over-filter the signal.

    See Also
    --------
    estimate_bandpass : Pass-band estimator used when filtering.
    estimate_bandpass_centroid : Alternative estimator; its output can
        be used to filter the signal manually.
    isolate_echoes, gate_echo : Used to isolate the strongest echo.
    """

    # Ensure the signal is a numpy array (no copy if already an ndarray)
    signal = np.asarray(raw_signal)

    # Validate inputs
    _validate_process_signal(roi_samples, len(signal), sampling_rate_hz)

    # Crop signal to region of interest
    roi_signal = signal[roi_samples[0] : roi_samples[1]]

    # Apply detrending
    if auto_detrend:
        roi_signal = detrend(roi_signal, type="linear")
    else:
        roi_signal = detrend(roi_signal, type="constant")  # Only remove DC offset if not detrending linearly

    # Apply filter if requested, with the pass band estimated from the
    # strongest echo only (see Notes)
    filter_params = None
    if apply_filter:
        strongest_echo = isolate_echoes(roi_signal, n_echoes=1)[0]
        reference_echo = gate_echo(roi_signal, strongest_echo)
        filter_params = estimate_bandpass(reference_echo, sampling_rate_hz)

        # Calculate Nyquist frequency
        nyquist = 0.5 * sampling_rate_hz

        # Normalize filter frequencies
        low_normalized = filter_params["lowcut"] / nyquist
        high_normalized = filter_params["highcut"] / nyquist

        # Design zero-phase Butterworth bandpass filter
        sos = butter(
            N=filter_params["order"],
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
                f"{filter_params['order']}; it must exceed {padlen} samples. "
                "Use a wider ROI."
            )

        # Apply zero-phase filtering (forward and backward)
        # this results in zero phase distortion, which is important
        # for preserving the timing and shape of the ultrasound pulses.
        roi_signal = sosfiltfilt(sos, roi_signal)

    if return_filter_params:
        return roi_signal, filter_params
    return roi_signal


def detect_roi(
    signal: npt.ArrayLike,
    threshold: float = 0.1,
    pad_samples: int = 0,
) -> tuple[int, int]:
    """
    Detect the region of interest of a pulse from its analytic-signal
    envelope (Hilbert transform).

    The envelope ``|hilbert(signal)|`` is a smooth measure of the
    instantaneous amplitude. The region of interest is taken as the
    first and last samples whose envelope reaches a given fraction of
    the envelope peak, optionally widened by ``pad_samples``. The
    returned tuple is directly usable as the ``roi_samples`` argument
    of :func:`process_signal`.

    Parameters
    ----------
    signal : array-like
        The raw, full-length signal to search for a pulse.
    threshold : float, optional
        Onset level as a fraction of the envelope peak amplitude, in
        the open interval (0, 1). The region of interest spans the
        samples whose envelope is at least ``threshold`` times the
        peak. By default 0.1 (10% of the peak).
    pad_samples : int, optional
        Number of samples added to each side of the detected region,
        clipped to the signal bounds. Useful to include the pulse
        rise/decay tails or to provide padding for a later zero-phase
        filter. By default 0.

    Returns
    -------
    tuple of int
        The (start, end) sample indices of the region of interest, with
        ``end`` exclusive, following the half-open slicing convention of
        :func:`process_signal`.

    Raises
    ------
    ValueError
        If the inputs are malformed (see :func:`_validate_detect_roi`)
        or if the envelope is identically zero (no pulse to detect).

    Notes
    -----
    The fraction-of-peak threshold is robust for high signal-to-noise
    pulse-echo records. For noisy data, pre-filtering the signal (or
    using a noise-relative threshold) gives a more reliable onset.

    See Also
    --------
    trigger_sta_lta : Energy-ratio onset detector for noisier traces.
    process_signal : Consumer of the returned ``roi_samples``.
    """

    signal = np.asarray(signal)
    _validate_detect_roi(signal, threshold, pad_samples)

    # Instantaneous amplitude (envelope) via the analytic signal
    envelope = np.abs(hilbert(signal))

    peak = envelope.max()
    if peak == 0:
        raise ValueError(
            "signal envelope is identically zero; no pulse to detect."
        )

    # Samples reaching the onset level relative to the envelope peak
    level = threshold * peak
    above = np.flatnonzero(envelope >= level)

    # Half-open interval, widened and clipped to the signal bounds
    start = max(0, int(above[0]) - pad_samples)
    end = min(signal.shape[0], int(above[-1]) + 1 + pad_samples)

    return start, end


def isolate_echoes(
    signal: npt.ArrayLike,
    n_echoes: int = 2,
    threshold: float = 0.1,
    pad: float = 0.2,
) -> list[tuple[int, int]]:
    """
    Locate the sample bounds of individual echoes within a region of
    interest from its analytic-signal envelope (Hilbert transform).

    The ``n_echoes`` most prominent envelope peaks are taken as the
    echo centres. Consecutive echoes are separated at the envelope
    minimum (valley) between their peaks. Each echo then spans the
    samples around its peak whose envelope reaches ``threshold`` times
    that echo's own peak, widened by ``pad`` without crossing the
    neighbouring valleys. The returned bounds are directly usable by
    :func:`gate_echo`.

    Parameters
    ----------
    signal : array-like
        The signal containing the echoes, typically the output of
        :func:`process_signal` (cropped, detrended and filtered).
    n_echoes : int, optional
        Number of echoes to isolate, by default 2.
    threshold : float, optional
        Fraction in the open interval (0, 1), by default 0.1. It sets
        both the minimum envelope-peak prominence for an echo, relative
        to the largest envelope peak, and the level, relative to each
        echo's own peak, that defines the echo's extent. Lower it if a
        strongly attenuated echo is not detected.
    pad : float, optional
        Fraction of each echo's width added to each side of it, by
        default 0.2. The padding is meant to hold the taper of the
        Tukey window applied by :func:`gate_echo`, so that the taper
        does not attenuate the echo itself.

    Returns
    -------
    list of tuple of int
        The (start, end) sample indices of each echo in time order,
        with ``end`` exclusive (half-open slicing convention).

    Raises
    ------
    ValueError
        If the inputs are malformed (see
        :func:`_validate_isolate_echoes`), the envelope is identically
        zero, or fewer than ``n_echoes`` envelope peaks exceed the
        prominence threshold.

    Warns
    -----
    UserWarning
        If the envelope valley between two consecutive echoes is not
        below ``threshold`` times the smaller of their peaks, i.e. the
        echoes overlap and their bounds are cut at the valley, which
        truncates them.

    See Also
    --------
    detect_roi : Single-pulse version of the same envelope criterion.
    gate_echo : Applies a Tukey window to the returned bounds.
    """

    signal = np.asarray(signal)
    _validate_isolate_echoes(signal, n_echoes, threshold, pad)

    # Instantaneous amplitude (envelope) via the analytic signal
    envelope = np.abs(hilbert(signal))
    if envelope.max() == 0:
        raise ValueError("signal envelope is identically zero; no echoes.")

    # The n most prominent envelope peaks, in time order
    peaks, properties = find_peaks(
        envelope, prominence=threshold * envelope.max()
    )
    if peaks.size < n_echoes:
        raise ValueError(
            f"found {peaks.size} envelope peak(s) above the prominence "
            f"threshold, fewer than n_echoes={n_echoes}. Lower threshold "
            "or check the region of interest."
        )
    strongest = np.argsort(properties["prominences"])[-n_echoes:]
    peaks = np.sort(peaks[strongest])

    # Envelope minimum between consecutive echoes; with the record edges,
    # these limit how far each echo may extend
    valleys = [
        left + int(np.argmin(envelope[left:right]))
        for left, right in zip(peaks[:-1], peaks[1:])
    ]
    for valley, left, right in zip(valleys, peaks[:-1], peaks[1:]):
        if envelope[valley] >= threshold * min(envelope[left], envelope[right]):
            warnings.warn(
                f"Echoes at samples {left} and {right} overlap: the "
                "envelope between them does not fall below threshold "
                "times the smaller peak, so the gates are cut at the "
                "valley and truncate the echoes.",
                stacklevel=2,
            )
    limits = [0, *valleys, signal.shape[0]]

    bounds = []
    for k, peak in enumerate(peaks):
        low_limit, high_limit = limits[k], limits[k + 1]
        level = threshold * envelope[peak]

        # Last sample below the level before the peak and first after it
        below_left = np.flatnonzero(envelope[low_limit:peak] < level)
        below_right = np.flatnonzero(envelope[peak:high_limit] < level)
        if below_left.size:
            start = low_limit + below_left[-1] + 1
        else:
            start = low_limit
        if below_right.size:
            end = peak + below_right[0]
        else:
            end = high_limit

        # Widen for the window taper, without crossing the valleys
        padding = int(round(pad * (end - start)))
        start = max(low_limit, start - padding)
        end = min(high_limit, end + padding)
        bounds.append((int(start), int(end)))

    return bounds


def gate_echo(
    signal: npt.ArrayLike,
    bounds: tuple[int, int],
    tukey_alpha: float = 0.25,
) -> np.ndarray:
    """
    Isolate one echo by applying a Tukey window over its bounds and
    setting the rest of the signal to zero.

    The output keeps the length of the input, so the sample index (and
    hence the time) of the echo is preserved and the lags of a
    cross-correlation between two gated echoes map directly to time
    differences.

    Parameters
    ----------
    signal : array-like
        The signal containing the echo, typically the output of
        :func:`process_signal`.
    bounds : tuple of int
        The (start, end) sample indices of the echo, ``end`` exclusive,
        e.g. one element of the list returned by
        :func:`isolate_echoes`. Must satisfy
        ``0 <= start < end <= len(signal)``.
    tukey_alpha : float, optional
        Fraction of the window, in [0, 1], occupied by the cosine
        tapers, by default 0.25. 0 gives a rectangular window and 1 a
        Hann window. Keep the tapers within the padding added by
        :func:`isolate_echoes` (i.e. ``tukey_alpha / 2 <= pad /
        (1 + 2 * pad)`` on each side) so that the echo is not
        attenuated.

    Returns
    -------
    np.ndarray
        The gated signal, the same length as ``signal``.

    Raises
    ------
    ValueError
        If the inputs are malformed (see :func:`_validate_gate_echo`).

    See Also
    --------
    isolate_echoes : Provides the echo bounds.
    """

    signal = np.asarray(signal, dtype=float)
    _validate_gate_echo(signal, bounds, tukey_alpha)

    start, end = bounds
    gated = np.zeros_like(signal)
    gated[start:end] = signal[start:end] * tukey(end - start, tukey_alpha)

    return gated


def estimate_bandpass(
    signal: npt.ArrayLike,
    sampling_rate_hz: float,
    margin: float = 0.2,
) -> dict:
    """
    Estimate band-pass cut-off frequencies from a signal's amplitude
    spectrum using a half-maximum (-6 dB) threshold.

    Parameters
    ----------
    signal : array-like
        The input signal, typically a single cropped and detrended
        echo (e.g. the output of :func:`gate_echo`). A record with
        several echoes has interference ripples in its spectrum that
        can truncate the estimated band.
    sampling_rate_hz : float
        The digitizer sampling rate in Hz, i.e. ``1 / dt`` where ``dt``
        is the time step between consecutive samples. Not the
        transducer's nominal centre frequency.
    margin : float, optional
        Fractional padding added to each side of the detected band,
        expressed as a fraction of the band width, by default 0.2.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'lowcut' : float
            Lower cut-off frequency in Hz, always > 0.
        - 'highcut' : float
            Upper cut-off frequency in Hz, always below the Nyquist
            frequency.
        - 'order' : int
            Suggested Butterworth filter order (fixed at 4).
        - 'peak_frequency' : float
            Frequency of the spectral peak in Hz.

    Raises
    ------
    ValueError
        If the inputs are malformed (see
        :func:`_validate_bandpass_inputs`) or the spectrum is zero.

    Warns
    -----
    UserWarning
        If the widened band extends to 0 Hz or to the Nyquist
        frequency and is clipped to the valid frequency range.

    Notes
    -----
    The band is the contiguous range of frequencies around the spectral
    peak whose amplitude is at least 50% (-6 dB) of the peak, the
    bandwidth convention of ASTM E1065 for ultrasonic transducers. The
    band edges are located by linear interpolation between the two
    frequency bins that straddle the threshold, so they are not limited
    to the FFT bin spacing. The edges are then widened by ``margin``
    and clipped to the range of positive frequencies below Nyquist.
    Secondary peaks separated from the main lobe by a dip below the
    threshold are ignored.

    Which to use
    ------------
    This is the estimator used by :func:`process_signal`. Because the
    band is tied to the main spectral lobe, it is insensitive to
    broadband noise as long as the noise stays below half the peak
    amplitude. Use :func:`estimate_bandpass_centroid` when a band
    derived from the spectral moments (centroid and spread) is
    preferred.
    """

    signal = np.asarray(signal, dtype=float)
    _validate_bandpass_inputs(signal, sampling_rate_hz)
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    freqs, spectrum = _positive_spectrum(signal, sampling_rate_hz)

    peak_index = int(np.argmax(spectrum))
    f_peak = freqs[peak_index]

    # -6 dB band around the peak, with interpolated edges
    threshold = 0.5 * spectrum[peak_index]
    lowcut, highcut = _band_edges(freqs, spectrum, peak_index, threshold)

    # add a safety margin
    bandwidth = highcut - lowcut
    lowcut -= margin * bandwidth
    highcut += margin * bandwidth

    lowcut, highcut = _clip_band(lowcut, highcut, freqs)

    return {
        "lowcut": lowcut,
        "highcut": highcut,
        "order": 4,
        "peak_frequency": f_peak
    }


def estimate_bandpass_centroid(
    signal: npt.ArrayLike,
    sampling_rate_hz: float,
    sigma_mult: float = 3.0,
    floor_db: float = -20.0,
) -> dict:
    """
    Estimate band-pass cut-off frequencies from the spectral centroid
    and spectral spread of a signal.

    Parameters
    ----------
    signal : array-like
        The input signal, typically a single cropped and detrended
        echo (e.g. the output of :func:`gate_echo`). A record with
        several echoes has interference ripples in its spectrum that
        can truncate the estimated band.
    sampling_rate_hz : float
        The digitizer sampling rate in Hz, i.e. ``1 / dt`` where ``dt``
        is the time step between consecutive samples. Not the
        transducer's nominal centre frequency.
    sigma_mult : float, optional
        Number of spectral standard deviations on each side of the
        centroid used to set the band edges, by default 3.0. For a
        Gaussian spectrum, +/-3 sigma of the power spectrum reaches
        about -20 dB in power (-10 dB in amplitude).
    floor_db : float, optional
        Power level, in dB relative to the spectral peak (negative),
        that bounds the contiguous band around the peak over which the
        spectral moments are computed, by default -20.0. Excluding the
        spectrum below this floor stops broadband noise from biasing
        the centroid and inflating the spread.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'lowcut' : float
            Lower cut-off frequency in Hz, always > 0.
        - 'highcut' : float
            Upper cut-off frequency in Hz, always below the Nyquist
            frequency.
        - 'order' : int
            Suggested Butterworth filter order (fixed at 4).
        - 'center_frequency' : float
            Power-weighted spectral centroid in Hz.
        - 'bandwidth_sigma' : float
            Power-weighted spectral standard deviation (spread) in Hz.

    Raises
    ------
    ValueError
        If the inputs are malformed (see
        :func:`_validate_bandpass_inputs`), ``floor_db`` is not
        negative, or the spectrum is zero.

    Warns
    -----
    UserWarning
        If the band extends to 0 Hz or to the Nyquist frequency and is
        clipped to the valid frequency range.

    Notes
    -----
    A Hann window is applied before the FFT to reduce spectral
    leakage. The centroid and spread are the first moment and the
    square root of the second central moment of the power spectrum
    ``|X(f)|**2``, as in the centroid frequency-shift method of Quan &
    Harris (1997), restricted to the contiguous band around the peak
    whose power is above ``floor_db``. For a Gaussian-modulated pulse
    whose amplitude spectrum has standard deviation ``s``, the
    power-weighted spread is ``s / sqrt(2)`` (slightly broadened by the
    Hann window).

    Which to use
    ------------
    Use this estimator when a band derived from the spectral moments is
    preferred, e.g. for asymmetric spectra where the centroid is a
    better measure of the dominant frequency than the peak. The -6 dB
    :func:`estimate_bandpass` is the estimator used by
    :func:`process_signal`.

    References
    ----------
    Quan, Y., & Harris, J. M. (1997). Seismic attenuation tomography
    using the frequency shift method. Geophysics, 62(3), 895-905.
    """

    signal = np.asarray(signal, dtype=float)
    _validate_bandpass_inputs(signal, sampling_rate_hz)
    if floor_db >= 0:
        raise ValueError("floor_db must be negative.")

    # Window to reduce spectral leakage
    window = np.hanning(signal.shape[0])
    freqs, spectrum = _positive_spectrum(signal * window, sampling_rate_hz)
    power = spectrum**2

    # Restrict the moments to the contiguous band above the power floor
    peak_index = int(np.argmax(power))
    floor = power[peak_index] * 10 ** (floor_db / 10)
    low_index, high_index = _contiguous_band(power, peak_index, floor)
    band_freqs = freqs[low_index : high_index + 1]
    band_power = power[low_index : high_index + 1]

    # Spectral centroid
    fc = np.sum(band_freqs * band_power) / np.sum(band_power)

    # Spectral variance
    variance = np.sum(band_power * (band_freqs - fc)**2) / np.sum(band_power)
    sigma = np.sqrt(variance)

    # Band limits
    lowcut = fc - sigma_mult * sigma
    highcut = fc + sigma_mult * sigma

    lowcut, highcut = _clip_band(lowcut, highcut, freqs)

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


def _positive_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Amplitude spectrum of a real signal at the positive frequencies
    strictly below Nyquist (the DC and Nyquist bins are dropped).

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional real signal.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.

    Returns
    -------
    tuple of numpy.ndarray
        The frequencies in Hz and the amplitude spectrum ``|X(f)|``.

    Raises
    ------
    ValueError
        If the spectrum is identically zero.
    """

    freqs = rfftfreq(signal.shape[0], 1 / sampling_rate_hz)
    spectrum = np.abs(rfft(signal))

    keep = (freqs > 0) & (freqs < 0.5 * sampling_rate_hz)
    freqs = freqs[keep]
    spectrum = spectrum[keep]

    if spectrum.max() == 0:
        raise ValueError("signal spectrum is identically zero.")

    return freqs, spectrum


def _contiguous_band(
    spectrum: np.ndarray,
    peak_index: int,
    threshold: float,
) -> tuple[int, int]:
    """
    Indices of the contiguous run of bins around ``peak_index`` whose
    spectrum is at or above ``threshold``.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Non-negative spectrum (amplitude or power).
    peak_index : int
        Index of the spectral peak.
    threshold : float
        Level that bounds the band.

    Returns
    -------
    tuple of int
        The (low, high) bin indices of the band, both inclusive.
    """

    below_left = np.flatnonzero(spectrum[:peak_index] < threshold)
    below_right = np.flatnonzero(spectrum[peak_index:] < threshold)

    low_index = below_left[-1] + 1 if below_left.size else 0
    if below_right.size:
        high_index = peak_index + below_right[0] - 1
    else:
        high_index = spectrum.shape[0] - 1

    return int(low_index), int(high_index)


def _band_edges(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peak_index: int,
    threshold: float,
) -> tuple[float, float]:
    """
    Frequencies at which the spectrum crosses ``threshold`` on either
    side of the peak, linearly interpolated between the straddling bins.

    Where the band reaches the end of the spectrum without a crossing,
    the outermost frequency is returned for that edge.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frequencies in Hz, increasing.
    spectrum : numpy.ndarray
        Spectrum evaluated at ``freqs``.
    peak_index : int
        Index of the spectral peak.
    threshold : float
        Level that defines the band edges.

    Returns
    -------
    tuple of float
        The lower and upper edge frequencies in Hz.
    """

    low_index, high_index = _contiguous_band(spectrum, peak_index, threshold)

    lower_edge = freqs[low_index]
    if low_index > 0:
        # spectrum rises through the threshold from low_index-1 to low_index
        lower_edge = np.interp(
            threshold,
            spectrum[low_index - 1 : low_index + 1],
            freqs[low_index - 1 : low_index + 1],
        )

    upper_edge = freqs[high_index]
    if high_index < spectrum.shape[0] - 1:
        # spectrum falls through the threshold from high_index to
        # high_index+1; np.interp needs increasing sample points
        upper_edge = np.interp(
            threshold,
            spectrum[high_index : high_index + 2][::-1],
            freqs[high_index : high_index + 2][::-1],
        )

    return float(lower_edge), float(upper_edge)


def _clip_band(
    lowcut: float,
    highcut: float,
    freqs: np.ndarray,
) -> tuple[float, float]:
    """
    Clip band-pass cut-offs to the positive frequencies below Nyquist
    covered by ``freqs``, warning when clipping occurs.

    Parameters
    ----------
    lowcut, highcut : float
        Estimated cut-off frequencies in Hz.
    freqs : numpy.ndarray
        Positive frequencies below Nyquist, increasing.

    Returns
    -------
    tuple of float
        The clipped (lowcut, highcut) in Hz.
    """

    if lowcut < freqs[0]:
        warnings.warn(
            f"Estimated lowcut ({lowcut:.4g} Hz) is below the lowest "
            f"positive frequency; clipped to {freqs[0]:.4g} Hz.",
            stacklevel=3,
        )
        lowcut = freqs[0]
    if highcut > freqs[-1]:
        warnings.warn(
            f"Estimated highcut ({highcut:.4g} Hz) reaches the Nyquist "
            f"frequency; clipped to {freqs[-1]:.4g} Hz.",
            stacklevel=3,
        )
        highcut = freqs[-1]

    return float(lowcut), float(highcut)


def _validate_bandpass_inputs(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> None:
    """
    Validate the inputs shared by :func:`estimate_bandpass` and
    :func:`estimate_bandpass_centroid`.

    Parameters
    ----------
    signal : numpy.ndarray
        The signal, as an array.
    sampling_rate_hz : float
        The sampling rate of the signal in Hz.

    Raises
    ------
    ValueError
        If any input is malformed.
    """

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be a positive number.")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.shape[0] < 3:
        raise ValueError("signal must contain at least 3 samples.")


def _validate_process_signal(
    roi_samples: tuple[int, int],
    signal_length: int,
    sampling_rate_hz: float,
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


def _validate_detect_roi(
    signal: np.ndarray,
    threshold: float,
    pad_samples: int,
) -> None:
    """
    Validate the inputs of :func:`detect_roi`.

    Parameters
    ----------
    signal : numpy.ndarray
        The signal to search, as an array.
    threshold : float
        Onset level as a fraction of the envelope peak.
    pad_samples : int
        Number of samples added to each side of the detected region.

    Raises
    ------
    ValueError
        If any input is malformed.
    """

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in the open interval (0, 1).")
    if not isinstance(pad_samples, (int, np.integer)) or pad_samples < 0:
        raise ValueError("pad_samples must be a non-negative integer.")


def _validate_isolate_echoes(
    signal: np.ndarray,
    n_echoes: int,
    threshold: float,
    pad: float,
) -> None:
    """
    Validate the inputs of :func:`isolate_echoes`.

    Parameters
    ----------
    signal : numpy.ndarray
        The signal to search, as an array.
    n_echoes : int
        Number of echoes to isolate.
    threshold : float
        Prominence and extent level as a fraction of the envelope peak.
    pad : float
        Fraction of each echo's width added to each side of it.

    Raises
    ------
    ValueError
        If any input is malformed.
    """

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("signal must not be empty.")
    if not isinstance(n_echoes, (int, np.integer)) or n_echoes < 1:
        raise ValueError("n_echoes must be a positive integer.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in the open interval (0, 1).")
    if pad < 0:
        raise ValueError("pad must be non-negative.")


def _validate_gate_echo(
    signal: np.ndarray,
    bounds: tuple[int, int],
    tukey_alpha: float,
) -> None:
    """
    Validate the inputs of :func:`gate_echo`.

    Parameters
    ----------
    signal : numpy.ndarray
        The signal containing the echo, as an array.
    bounds : tuple of int
        The (start, end) sample indices of the echo.
    tukey_alpha : float
        Taper fraction of the Tukey window.

    Raises
    ------
    ValueError
        If any input is malformed.
    """

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if len(bounds) != 2:
        raise ValueError("bounds must be a (start, end) tuple of length 2.")
    start, end = bounds
    if not (isinstance(start, (int, np.integer))
            and isinstance(end, (int, np.integer))):
        raise ValueError("bounds start and end must be integers.")
    if start < 0 or end <= start or end > signal.shape[0]:
        raise ValueError(
            "bounds must satisfy 0 <= start < end <= len(signal)."
        )
    if not 0 <= tukey_alpha <= 1:
        raise ValueError("tukey_alpha must lie in the closed interval [0, 1].")


# End of file
