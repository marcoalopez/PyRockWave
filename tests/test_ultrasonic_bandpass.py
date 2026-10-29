"""Validation of the band-pass estimators in ultrasonic.py.

Both ``estimate_bandpass`` (half-maximum / -6 dB band) and
``estimate_bandpass_centroid`` (spectral centroid +/- N*sigma) are exercised
against synthetic pulses whose spectra are known analytically.

The test signal is a Gaussian-modulated cosine (a "tone burst"), the standard
idealisation of a pulse-echo ultrasonic wavelet:

    s(t) = exp(-(t - t0)^2 / (2 tau^2)) * cos(2 pi f0 t)

Its magnitude spectrum is a Gaussian centred on the carrier frequency f0 with
standard deviation (in Hz)

    sigma_f = 1 / (2 pi tau).

This gives closed-form targets:
  * the spectral peak / centroid must recover f0;
  * the centroid spread must recover sigma_f (broadened slightly by the Hann
    window used inside estimate_bandpass_centroid);
  * the -6 dB half-width must be sigma_f * sqrt(2 ln 2).

Two convention-independent behavioural anchors are also checked:
  (A) the recovered band must bracket the carrier (lowcut < f0 < highcut);
  (B) a temporally narrower pulse (broader spectrum) must yield a wider band.

Run standalone (no pytest dependency):
    python tests/test_ultrasonic_bandpass.py
"""
import os
import sys
import warnings

import numpy as np
from numpy.polynomial import Polynomial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.signal import correlate, correlation_lags, hilbert  # noqa: E402

from pyrockwave.ultrasonic import (  # noqa: E402
    detect_roi,
    estimate_bandpass,
    estimate_bandpass_centroid,
    estimate_peak,
    estimate_sampling_rate,
    estimate_travel_time,
    gate_echo,
    isolate_echoes,
    process_signal,
)


# --------------------------------------------------------------------------
# Synthetic tone-burst generator
# --------------------------------------------------------------------------
def gaussian_tone_burst(f0, tau, sampling_rate_hz, n_samples):
    """Gaussian-modulated cosine centred in the record.

    Returns the signal and its analytic spectral standard deviation
    sigma_f = 1 / (2 pi tau) in Hz.
    """
    dt = 1.0 / sampling_rate_hz
    t = np.arange(n_samples) * dt
    t0 = 0.5 * n_samples * dt
    envelope = np.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))
    signal = envelope * np.cos(2.0 * np.pi * f0 * t)
    sigma_f = 1.0 / (2.0 * np.pi * tau)
    return signal, sigma_f


# Common acquisition settings: 100 MHz sampling, 5 MHz carrier.
SAMPLING_RATE_HZ = 100e6
N_SAMPLES = 4096
F0 = 5e6
TAU = 2e-6  # -> sigma_f ~ 79.6 kHz


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_estimate_bandpass_recovers_carrier():
    """Peak frequency ~ f0 and the band brackets the carrier."""
    signal, sigma_f = gaussian_tone_burst(F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)
    res = estimate_bandpass(signal, SAMPLING_RATE_HZ)

    assert np.isclose(res["peak_frequency"], F0, rtol=0.01)
    assert res["lowcut"] < F0 < res["highcut"]
    assert res["order"] == 4

    # -6 dB half-width of a Gaussian is sigma_f * sqrt(2 ln 2); the reported
    # band adds the default 20% margin, so the half-width must be at least the
    # bare -6 dB value and of the right order of magnitude.
    half_width = 0.5 * (res["highcut"] - res["lowcut"])
    minus6db_half = sigma_f * np.sqrt(2.0 * np.log(2.0))
    assert minus6db_half < half_width < 4.0 * minus6db_half


def test_estimate_bandpass_centroid_recovers_carrier_and_sigma():
    """Centroid ~ f0 and spread ~ analytic power-spectrum sigma."""
    signal, sigma_f = gaussian_tone_burst(F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)
    res = estimate_bandpass_centroid(signal, SAMPLING_RATE_HZ)

    assert np.isclose(res["center_frequency"], F0, rtol=0.02)
    assert res["lowcut"] < F0 < res["highcut"]
    assert res["order"] == 4

    # The power spectrum |X|^2 of a Gaussian amplitude spectrum with standard
    # deviation sigma_f is Gaussian with standard deviation sigma_f / sqrt(2).
    assert np.isclose(res["bandwidth_sigma"], sigma_f / np.sqrt(2.0), rtol=0.1)


def test_band_widens_for_narrower_pulse():
    """A temporally narrower pulse has a broader spectrum -> wider band.

    This is convention-independent: it holds for both estimators regardless of
    threshold or windowing details.
    """
    wide_pulse, _ = gaussian_tone_burst(F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)
    narrow_pulse, _ = gaussian_tone_burst(
        F0, 0.5 * TAU, SAMPLING_RATE_HZ, N_SAMPLES
    )

    bp_wide = estimate_bandpass(wide_pulse, SAMPLING_RATE_HZ)
    bp_narrow = estimate_bandpass(narrow_pulse, SAMPLING_RATE_HZ)
    assert (bp_narrow["highcut"] - bp_narrow["lowcut"]
            > bp_wide["highcut"] - bp_wide["lowcut"])

    ct_wide = estimate_bandpass_centroid(wide_pulse, SAMPLING_RATE_HZ)
    ct_narrow = estimate_bandpass_centroid(narrow_pulse, SAMPLING_RATE_HZ)
    assert ct_narrow["bandwidth_sigma"] > ct_wide["bandwidth_sigma"]


def test_carrier_shift_tracks_frequency():
    """Doubling the carrier doubles the recovered peak/centroid frequency."""
    low, _ = gaussian_tone_burst(F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)
    high, _ = gaussian_tone_burst(2.0 * F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)

    assert np.isclose(
        estimate_bandpass(high, SAMPLING_RATE_HZ)["peak_frequency"],
        2.0 * estimate_bandpass(low, SAMPLING_RATE_HZ)["peak_frequency"],
        rtol=0.02,
    )
    assert np.isclose(
        estimate_bandpass_centroid(high, SAMPLING_RATE_HZ)["center_frequency"],
        2.0 * estimate_bandpass_centroid(low, SAMPLING_RATE_HZ)["center_frequency"],
        rtol=0.02,
    )


# Short-record settings (a typical cropped ROI): 1024 samples -> ~98 kHz bins,
# a shorter pulse with a broader spectrum (sigma_f ~ 530 kHz).
N_SHORT = 1024
TAU_SHORT = 0.3e-6


def add_white_noise(signal, snr_db, seed=0):
    """Add Gaussian white noise at a given SNR (signal std / noise std)."""
    rng = np.random.default_rng(seed)
    noise_std = np.std(signal) / 10 ** (snr_db / 20)
    return signal + rng.normal(0.0, noise_std, signal.shape[0])


def test_estimate_bandpass_edges_match_analytic_minus6db():
    """With margin=0 the interpolated edges match the analytic -6 dB points
    to well within one FFT bin."""
    signal, sigma_f = gaussian_tone_burst(
        F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT
    )
    res = estimate_bandpass(signal, SAMPLING_RATE_HZ, margin=0.0)

    half_width = sigma_f * np.sqrt(2.0 * np.log(2.0))
    bin_width = SAMPLING_RATE_HZ / N_SHORT
    assert abs(res["lowcut"] - (F0 - half_width)) < 0.1 * bin_width
    assert abs(res["highcut"] - (F0 + half_width)) < 0.1 * bin_width


def test_estimate_bandpass_ignores_secondary_peak():
    """A separate spectral peak above -6 dB must not widen the band: only
    the contiguous lobe around the main peak defines it."""
    signal, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    second, _ = gaussian_tone_burst(
        3.0 * F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT
    )
    res = estimate_bandpass(signal + 0.6 * second, SAMPLING_RATE_HZ)

    assert res["lowcut"] < F0 < res["highcut"] < 2.0 * F0


def test_estimators_robust_to_white_noise():
    """At 20 dB SNR both estimators stay close to their noise-free values.

    Broadband noise used to dominate the whole-spectrum centroid moments
    (centroid pulled towards Nyquist, spread inflated by ~20x).
    """
    clean, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    noisy = add_white_noise(clean, snr_db=20.0)

    bp_clean = estimate_bandpass(clean, SAMPLING_RATE_HZ)
    bp_noisy = estimate_bandpass(noisy, SAMPLING_RATE_HZ)
    assert np.isclose(bp_noisy["lowcut"], bp_clean["lowcut"], rtol=0.05)
    assert np.isclose(bp_noisy["highcut"], bp_clean["highcut"], rtol=0.05)

    ct_clean = estimate_bandpass_centroid(clean, SAMPLING_RATE_HZ)
    ct_noisy = estimate_bandpass_centroid(noisy, SAMPLING_RATE_HZ)
    assert np.isclose(ct_noisy["center_frequency"], F0, rtol=0.02)
    assert np.isclose(
        ct_noisy["bandwidth_sigma"], ct_clean["bandwidth_sigma"], rtol=0.1
    )


def test_estimators_clip_band_to_valid_range():
    """A broadband pulse near DC gives a band that would extend below 0 Hz;
    it is clipped to (0, Nyquist) with a warning, never returned negative."""
    dt = 1.0 / SAMPLING_RATE_HZ
    t = np.arange(N_SHORT) * dt
    t0 = 0.5 * N_SHORT * dt
    # Very short unmodulated Gaussian: spectrum peaks at the lowest bin.
    signal = np.exp(-((t - t0) ** 2) / (2.0 * (0.02e-6) ** 2))
    nyquist = 0.5 * SAMPLING_RATE_HZ

    for estimator in (estimate_bandpass, estimate_bandpass_centroid):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = estimator(signal, SAMPLING_RATE_HZ)
        assert 0.0 < res["lowcut"] < res["highcut"] < nyquist
        assert any("lowcut" in str(w.message) for w in caught)


def test_estimators_reject_bad_inputs():
    """Malformed inputs and an all-zero signal raise ValueError."""
    signal, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    bad_calls = [
        lambda: estimate_bandpass(signal, 0.0),
        lambda: estimate_bandpass(signal.reshape(2, -1), SAMPLING_RATE_HZ),
        lambda: estimate_bandpass(np.zeros(N_SHORT), SAMPLING_RATE_HZ),
        lambda: estimate_bandpass(signal, SAMPLING_RATE_HZ, margin=-0.1),
        lambda: estimate_bandpass_centroid(signal, -1.0),
        lambda: estimate_bandpass_centroid(np.zeros(N_SHORT), SAMPLING_RATE_HZ),
        lambda: estimate_bandpass_centroid(
            signal, SAMPLING_RATE_HZ, floor_db=3.0
        ),
    ]
    for call in bad_calls:
        try:
            call()
            raise AssertionError("bad input should have raised")
        except ValueError:
            pass


def test_process_signal_without_filter():
    """apply_filter=False only crops and detrends; no filter params."""
    signal, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    shifted = signal + 3.0  # DC offset removed by the detrend

    out, params = process_signal(
        shifted,
        (0, N_SHORT),
        SAMPLING_RATE_HZ,
        auto_detrend=False,
        return_filter_params=True,
    )
    assert params is None
    assert np.allclose(out, signal - signal.mean(), atol=1e-6)


def test_process_signal_auto_filter_reduces_noise():
    """apply_filter=True estimates the band from the ROI itself and removes
    out-of-band noise while keeping the pulse (zero-phase: no time shift)."""
    clean, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    noisy = add_white_noise(clean, snr_db=10.0)

    out, params = process_signal(
        noisy,
        (0, N_SHORT),
        SAMPLING_RATE_HZ,
        apply_filter=True,
        return_filter_params=True,
    )
    assert params["lowcut"] < F0 < params["highcut"]
    assert np.std(out - clean) < 0.5 * np.std(noisy - clean)
    assert np.argmax(np.abs(out)) == np.argmax(np.abs(clean))

    # Default return is the signal alone.
    only_signal = process_signal(
        noisy, (0, N_SHORT), SAMPLING_RATE_HZ, apply_filter=True
    )
    assert np.allclose(only_signal, out)


def embed_burst(center_index, tau, f0, sampling_rate_hz, n_samples):
    """A tone burst centred on ``center_index`` within a longer record of
    near-silence (so the onset is well separated from the edges)."""
    dt = 1.0 / sampling_rate_hz
    t = np.arange(n_samples) * dt
    t0 = center_index * dt
    envelope = np.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))
    return envelope * np.cos(2.0 * np.pi * f0 * t)


def test_detect_roi_brackets_pulse():
    """The detected region brackets the burst and sits inside the record."""
    n = 8192
    center = 5000
    signal = embed_burst(center, TAU, F0, SAMPLING_RATE_HZ, n)

    start, end = detect_roi(signal)
    assert 0 < start < center < end < n
    # The envelope peak (the burst centre) must lie inside the region.
    assert start <= np.argmax(np.abs(signal)) < end


def test_detect_roi_threshold_and_padding():
    """Lower threshold widens the region; padding widens it further."""
    n = 8192
    signal = embed_burst(5000, TAU, F0, SAMPLING_RATE_HZ, n)

    tight = detect_roi(signal, threshold=0.5)
    loose = detect_roi(signal, threshold=0.05)
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])

    start, end = detect_roi(signal, threshold=0.5)
    p_start, p_end = detect_roi(signal, threshold=0.5, pad_samples=200)
    assert p_start == start - 200
    assert p_end == end + 200


def test_detect_roi_feeds_process_signal():
    """The returned tuple is directly usable as process_signal's ROI."""
    n = 8192
    signal = embed_burst(5000, TAU, F0, SAMPLING_RATE_HZ, n)

    roi = detect_roi(signal, pad_samples=100)
    out = process_signal(signal, roi, SAMPLING_RATE_HZ)
    assert out.shape[0] == roi[1] - roi[0]


def test_detect_roi_rejects_bad_inputs():
    """Malformed inputs and a flat signal raise ValueError."""
    signal = embed_burst(5000, TAU, F0, SAMPLING_RATE_HZ, 8192)

    for bad in (0.0, 1.0, -0.1):
        try:
            detect_roi(signal, threshold=bad)
            raise AssertionError(f"threshold={bad} should have raised")
        except ValueError:
            pass

    try:
        detect_roi(signal, pad_samples=-1)
        raise AssertionError("negative pad_samples should have raised")
    except ValueError:
        pass

    try:
        detect_roi(np.zeros(1000))
        raise AssertionError("all-zero signal should have raised")
    except ValueError:
        pass


# Two-echo pulse-echo record: 1 GS/s digitizer, 50 MHz transducer. The second
# echo is weaker and slightly down-shifted in frequency (attenuation), and the
# true delay is a fractional number of samples.
ECHO_FS = 1e9
ECHO_F0 = 50e6
ECHO_TAU = 25e-9
ECHO_N = 3000
ECHO_T1 = 0.6e-6
ECHO_DELAY = 1.2373e-6


def two_echo_record(delay=ECHO_DELAY, amplitude2=0.4, snr_db=None, seed=0):
    """Return a two-echo record and the true sample centres of the echoes."""
    t = np.arange(ECHO_N) / ECHO_FS

    def echo(tc, amplitude, f):
        envelope = np.exp(-((t - tc) ** 2) / (2.0 * ECHO_TAU ** 2))
        return amplitude * envelope * np.cos(2.0 * np.pi * f * (t - tc))

    t2 = ECHO_T1 + delay
    signal = echo(ECHO_T1, 1.0, ECHO_F0) + echo(t2, amplitude2, 0.92 * ECHO_F0)
    if snr_db is not None:
        signal = add_white_noise(signal, snr_db, seed=seed)
    return signal, (ECHO_T1 * ECHO_FS, t2 * ECHO_FS)


def cross_correlation_delay(echo1, echo2, expected_lag):
    """Sub-sample delay of echo2 relative to echo1 (in samples).

    Picks the cross-correlation lobe under the envelope maximum near
    ``expected_lag`` (avoids cycle skips), then refines with a 3-point
    parabola on |cc|.
    """
    cc = correlate(echo2, echo1, mode="full")
    lags = correlation_lags(echo2.shape[0], echo1.shape[0], mode="full")
    period = ECHO_FS / ECHO_F0

    near = np.flatnonzero(np.abs(lags - expected_lag) < 3 * period)
    centre = near[np.argmax(np.abs(hilbert(cc))[near])]
    lobe = np.flatnonzero(np.abs(lags - lags[centre]) <= period / 2)
    i = lobe[np.argmax(np.abs(cc[lobe]))]

    y0, y1, y2 = np.abs(cc[i - 1 : i + 2])
    return lags[i] + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2), np.sign(cc[i])


def test_isolate_echoes_brackets_each_echo():
    """Two disjoint, time-ordered bounds, each containing its echo centre."""
    signal, centres = two_echo_record()
    filtered = process_signal(signal, (0, ECHO_N), ECHO_FS, apply_filter=True)

    bounds = isolate_echoes(filtered)
    assert len(bounds) == 2
    (s1, e1), (s2, e2) = bounds
    assert 0 <= s1 < centres[0] < e1 <= s2 < centres[1] < e2 <= ECHO_N
    assert all(isinstance(i, int) for pair in bounds for i in pair)


def test_isolate_echoes_padding_widens_bounds():
    """pad=0 gives the bare threshold extent; padding widens each side."""
    signal, _ = two_echo_record()
    bare = isolate_echoes(signal, pad=0.0)
    padded = isolate_echoes(signal, pad=0.2)
    for (bs, be), (ps, pe) in zip(bare, padded):
        assert ps < bs and pe > be


def test_gate_echo_isolates_and_preserves_length():
    """Output keeps the input length, is zero outside the bounds, and
    leaves the echo centre untouched by the taper."""
    signal, centres = two_echo_record()
    bounds = isolate_echoes(signal)[0]
    gated = gate_echo(signal, bounds)

    start, end = bounds
    assert gated.shape == signal.shape
    assert np.all(gated[:start] == 0) and np.all(gated[end:] == 0)
    centre = int(round(centres[0]))
    assert np.isclose(gated[centre], signal[centre])


def test_isolate_and_gate_recover_delay_by_cross_correlation():
    """End-to-end: filter, isolate, gate, cross-correlate. The delay is
    recovered to a small fraction of a sample, also with noise and with an
    inverted second echo (polarity reported by the sign of the peak)."""
    for amplitude2 in (0.4, -0.4):
        for snr_db in (None, 20.0):
            signal, _ = two_echo_record(amplitude2=amplitude2, snr_db=snr_db)
            filtered = process_signal(
                signal, (0, ECHO_N), ECHO_FS, apply_filter=True
            )
            first, second = isolate_echoes(filtered)
            echo1 = gate_echo(filtered, first)
            echo2 = gate_echo(filtered, second)

            envelope = np.abs(hilbert(filtered))
            peak1 = first[0] + np.argmax(envelope[first[0] : first[1]])
            peak2 = second[0] + np.argmax(envelope[second[0] : second[1]])

            lag, sign = cross_correlation_delay(echo1, echo2, peak2 - peak1)
            assert abs(lag - ECHO_DELAY * ECHO_FS) < 0.1
            assert sign == np.sign(amplitude2)


def test_process_signal_band_robust_to_two_echo_interference():
    """With two echoes in the ROI the whole-record spectrum has ripples
    every 1/delay (~0.8 MHz here). The band must come from a single echo,
    so it must not collapse to the ripple period, and the filtered record
    must still show both echoes. Checked over many noise realisations at
    10 dB SNR, where the whole-record estimate used to fail."""
    for seed in range(20):
        signal, _ = two_echo_record(snr_db=10.0, seed=seed)
        filtered, params = process_signal(
            signal,
            (0, ECHO_N),
            ECHO_FS,
            apply_filter=True,
            return_filter_params=True,
        )
        assert params["lowcut"] < ECHO_F0 < params["highcut"]
        assert params["highcut"] - params["lowcut"] > 5e6
        assert len(isolate_echoes(filtered)) == 2


def test_estimate_sampling_rate_recovers_rate():
    """A uniform 1 GS/s time column gives 1e9 Hz, also when the times are
    rounded to picoseconds as in a text export."""
    t = np.arange(ECHO_N) / ECHO_FS + 3.2e-6  # non-zero start time
    assert np.isclose(estimate_sampling_rate(t), ECHO_FS, rtol=1e-12)
    assert np.isclose(
        estimate_sampling_rate(np.round(t, 12)), ECHO_FS, rtol=1e-6
    )


def test_estimate_sampling_rate_rounded_timestamps():
    """Times exported on a 1 ns grid while sitting almost half-way between
    grid points: a few round the other way, giving zero steps next to
    double steps. The rate stays exact and a warning names the cause."""
    rng = np.random.default_rng(0)
    i = np.arange(5000)
    t_true = 3e-6 + (i + 0.4997) / ECHO_FS + rng.normal(0.0, 1e-13, i.size)
    t_exported = np.char.mod("%.3f", t_true * 1e6).astype(float) * 1e-6
    assert np.any(np.diff(t_exported) == 0)  # the case being tested

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = estimate_sampling_rate(t_exported)
    assert np.isclose(rate, ECHO_FS, rtol=1e-9)
    messages = [str(w.message) for w in caught]
    assert any("rounded" in m for m in messages)


def test_estimate_sampling_rate_duplicated_rows():
    """Duplicated rows (zero steps with no double steps) are not counted
    as intervals, so the rate stays exact; the warning reports the rows."""
    t = np.arange(5000) / ECHO_FS
    t_duplicated = np.insert(t, [1000, 3000], t[[999, 2999]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rate = estimate_sampling_rate(t_duplicated)
    assert np.isclose(rate, ECHO_FS, rtol=1e-12)
    messages = [str(w.message) for w in caught]
    assert any("duplicated rows" in m and "1000" in m for m in messages)


def test_estimate_sampling_rate_rejects_non_uniform_time():
    """Missing samples, non-increasing times and bad inputs raise."""
    t = np.arange(ECHO_N) / ECHO_FS
    bad_calls = [
        lambda: estimate_sampling_rate(np.delete(t, 1000)),  # missing sample
        lambda: estimate_sampling_rate(t[::-1]),
        lambda: estimate_sampling_rate(np.zeros(10)),
        lambda: estimate_sampling_rate(t[:1]),
        lambda: estimate_sampling_rate(t.reshape(2, -1)),
        lambda: estimate_sampling_rate(np.append(t, np.nan)),
        lambda: estimate_sampling_rate(t, rtol=-1.0),
    ]
    for call in bad_calls:
        try:
            call()
            raise AssertionError("bad input should have raised")
        except ValueError:
            pass


def test_estimate_peak_interior_and_edge_maximum():
    """The maximum of a fitted quartic is found at its interior critical
    point (domain mapping of Polynomial.fit handled), and at the domain
    edge when the curve is monotonic there."""
    x =np.arange(1200.0, 1300.0, 10.0)
    peaked = Polynomial.fit(x, 5e3 - (x - 1237.3) ** 2, deg=4)
    peak_x, peak_y, at_boundary = estimate_peak(peaked, x)
    assert np.isclose(peak_x, 1237.3, atol=1e-6)
    assert np.isclose(peak_y, 5e3, rtol=1e-9)
    assert not at_boundary

    # Monotonic curve: maximum at the edge, flagged as such
    rising = Polynomial.fit(x, x - 1200.0, deg=4)
    edge_x, _, at_boundary = estimate_peak(rising, x)
    assert np.isclose(edge_x, x.max())
    assert at_boundary


def test_estimate_travel_time_recovers_delay():
    """Both models refine the central peak of the cross-correlation and
    recover the true two-way time (a fractional number of samples) to a
    small fraction of a sample, also with an inverted second echo and with
    noise; both report the polarity of the second echo."""
    true_us = ECHO_DELAY * 1e6
    sample_us = 1e6 / ECHO_FS
    for amplitude2 in (0.4, -0.4):
        for snr_db in (None, 20.0):
            signal, _ = two_echo_record(amplitude2=amplitude2, snr_db=snr_db)
            filtered = process_signal(
                signal, (0, ECHO_N), ECHO_FS, apply_filter=True
            )
            for fit in ("polynomial", "cosine"):
                time_us, details = estimate_travel_time(
                    filtered, ECHO_FS, fit=fit
                )
                assert abs(time_us - true_us) < 0.1 * sample_us
                assert np.sign(details["peak2_xy"][1]) == np.sign(amplitude2)


def test_estimate_travel_time_coarse_sampling():
    """At 4 samples per carrier period the central peak is too short for a
    quartic, which raises; the 3-point cosine fit still works."""
    signal, _ = two_echo_record()
    filtered = process_signal(signal, (0, ECHO_N), ECHO_FS, apply_filter=True)
    decimation = 5  # 1 GS/s -> 200 MS/s
    coarse = filtered[::decimation]
    coarse_fs = ECHO_FS / decimation

    try:
        estimate_travel_time(coarse, coarse_fs, fit="polynomial")
        raise AssertionError("too few lobe samples should have raised")
    except ValueError:
        pass

    time_us, _ = estimate_travel_time(coarse, coarse_fs, fit="cosine")
    assert abs(time_us - ECHO_DELAY * 1e6) < 0.1 * (1e6 / coarse_fs)


def test_estimate_travel_time_rejects_bad_inputs():
    """Unknown model, non-positive rate, or a single echo raise."""
    signal, _ = two_echo_record()
    single, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    bad_calls = [
        lambda: estimate_travel_time(signal, ECHO_FS, fit="spline"),
        lambda: estimate_travel_time(signal, 0.0),
        lambda: estimate_travel_time(single, SAMPLING_RATE_HZ),
    ]
    for call in bad_calls:
        try:
            call()
            raise AssertionError("bad input should have raised")
        except ValueError:
            pass


def test_isolate_echoes_warns_on_overlap():
    """Echoes closer than their extent are flagged; well-separated ones
    are not."""
    signal, _ = two_echo_record(delay=3.0 * ECHO_TAU, amplitude2=0.9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        isolate_echoes(signal)
    assert any("overlap" in str(w.message) for w in caught)

    signal, _ = two_echo_record()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        isolate_echoes(signal)
    assert not any("overlap" in str(w.message) for w in caught)


def test_isolate_and_gate_reject_bad_inputs():
    """Malformed inputs, a flat signal, or too few echoes raise ValueError."""
    signal, _ = two_echo_record()
    single, _ = gaussian_tone_burst(F0, TAU_SHORT, SAMPLING_RATE_HZ, N_SHORT)
    bad_calls = [
        lambda: isolate_echoes(signal, n_echoes=0),
        lambda: isolate_echoes(signal, threshold=1.0),
        lambda: isolate_echoes(signal, pad=-0.1),
        lambda: isolate_echoes(signal.reshape(2, -1)),
        lambda: isolate_echoes(np.zeros(100)),
        lambda: isolate_echoes(single, n_echoes=2),
        lambda: gate_echo(signal, (10, 5)),
        lambda: gate_echo(signal, (0, ECHO_N + 1)),
        lambda: gate_echo(signal, (0.0, 10.0)),
        lambda: gate_echo(signal, (0, 10), tukey_alpha=1.5),
    ]
    for call in bad_calls:
        try:
            call()
            raise AssertionError("bad input should have raised")
        except ValueError:
            pass


# --------------------------------------------------------------------------
# Standalone runner (mirrors the existing test style in this repo)
# --------------------------------------------------------------------------
def main():
    ok = True

    def check(name, func):
        nonlocal ok
        try:
            func()
            passed = True
        except AssertionError as exc:
            passed = False
            print(f"       -> {exc}")
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    check("estimate_bandpass recovers carrier",
          test_estimate_bandpass_recovers_carrier)
    check("estimate_bandpass_centroid recovers carrier and sigma",
          test_estimate_bandpass_centroid_recovers_carrier_and_sigma)
    check("band widens for narrower pulse",
          test_band_widens_for_narrower_pulse)
    check("carrier shift tracks frequency",
          test_carrier_shift_tracks_frequency)
    check("estimate_bandpass edges match analytic -6 dB",
          test_estimate_bandpass_edges_match_analytic_minus6db)
    check("estimate_bandpass ignores secondary peak",
          test_estimate_bandpass_ignores_secondary_peak)
    check("estimators robust to white noise",
          test_estimators_robust_to_white_noise)
    check("estimators clip band to valid range",
          test_estimators_clip_band_to_valid_range)
    check("estimators reject bad inputs",
          test_estimators_reject_bad_inputs)
    check("process_signal without filter",
          test_process_signal_without_filter)
    check("process_signal auto filter reduces noise",
          test_process_signal_auto_filter_reduces_noise)
    check("detect_roi brackets pulse",
          test_detect_roi_brackets_pulse)
    check("detect_roi threshold and padding",
          test_detect_roi_threshold_and_padding)
    check("detect_roi feeds process_signal",
          test_detect_roi_feeds_process_signal)
    check("detect_roi rejects bad inputs",
          test_detect_roi_rejects_bad_inputs)
    check("isolate_echoes brackets each echo",
          test_isolate_echoes_brackets_each_echo)
    check("isolate_echoes padding widens bounds",
          test_isolate_echoes_padding_widens_bounds)
    check("gate_echo isolates and preserves length",
          test_gate_echo_isolates_and_preserves_length)
    check("isolate + gate recover delay by cross-correlation",
          test_isolate_and_gate_recover_delay_by_cross_correlation)
    check("process_signal band robust to two-echo interference",
          test_process_signal_band_robust_to_two_echo_interference)
    check("estimate_sampling_rate recovers rate",
          test_estimate_sampling_rate_recovers_rate)
    check("estimate_sampling_rate rounded timestamps",
          test_estimate_sampling_rate_rounded_timestamps)
    check("estimate_sampling_rate duplicated rows",
          test_estimate_sampling_rate_duplicated_rows)
    check("estimate_sampling_rate rejects non-uniform time",
          test_estimate_sampling_rate_rejects_non_uniform_time)
    check("estimate_peak interior and edge maximum",
          test_estimate_peak_interior_and_edge_maximum)
    check("estimate_travel_time recovers delay",
          test_estimate_travel_time_recovers_delay)
    check("estimate_travel_time coarse sampling",
          test_estimate_travel_time_coarse_sampling)
    check("estimate_travel_time rejects bad inputs",
          test_estimate_travel_time_rejects_bad_inputs)
    check("isolate_echoes warns on overlap",
          test_isolate_echoes_warns_on_overlap)
    check("isolate_echoes / gate_echo reject bad inputs",
          test_isolate_and_gate_reject_bad_inputs)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
