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

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pyrockwave.ultrasonic import (  # noqa: E402
    detect_roi,
    estimate_bandpass,
    estimate_bandpass_centroid,
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
    """Centroid ~ f0 and spread ~ analytic sigma_f."""
    signal, sigma_f = gaussian_tone_burst(F0, TAU, SAMPLING_RATE_HZ, N_SAMPLES)
    res = estimate_bandpass_centroid(signal, SAMPLING_RATE_HZ)

    assert np.isclose(res["center_frequency"], F0, rtol=0.02)
    assert res["lowcut"] < F0 < res["highcut"]
    assert res["order"] == 4

    # The Hann window convolves the spectrum, so the measured sigma is at least
    # the analytic value and within a factor of a few of it.
    assert sigma_f <= res["bandwidth_sigma"] < 4.0 * sigma_f


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
    check("detect_roi brackets pulse",
          test_detect_roi_brackets_pulse)
    check("detect_roi threshold and padding",
          test_detect_roi_threshold_and_padding)
    check("detect_roi feeds process_signal",
          test_detect_roi_feeds_process_signal)
    check("detect_roi rejects bad inputs",
          test_detect_roi_rejects_bad_inputs)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
