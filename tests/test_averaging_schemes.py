"""Validation of the averaging schemes in averaging_schemes.py.

Checks invariants of Voigt/Reuss averaging that hold regardless of
implementation:

  (1) Single phase: averaging one phase (volume fraction 1, or a CPO
      with the identity orientation) returns the input tensor for both
      Voigt and Reuss schemes.
  (2) Two isotropic phases: the Voigt (arithmetic) and Reuss (harmonic)
      averages of the bulk and shear moduli match their closed-form
      values, and Voigt >= Reuss (classical bounds).
  (3) Isotropic invariance: an isotropic tensor is unchanged by CPO
      averaging with arbitrary orientations, for both schemes.
  (4) Single-orientation CPO: with one orientation the Voigt and Reuss
      CPO averages must coincide (both equal the rotated tensor), and
      the average preserves the Voigt bulk modulus (a rotational
      invariant).
  (5) Fractions that do not sum to 1 are normalised with a warning.

Plus the documented input-validation behaviour.

Run standalone (no pytest dependency):
    python tests/test_averaging_schemes.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

from pyrockwave.averaging_schemes import (
    voigt_volume_weighted_average,
    reuss_volume_weighted_average,
    voigt_CPO_weighted_average,
    reuss_CPO_weighted_average,
)

# Olivine at ambient conditions (Abramson et al., 1997), GPa
OLIVINE = np.array(
    [
        [320.5, 68.15, 71.6, 0.0, 0.0, 0.0],
        [68.15, 196.5, 76.8, 0.0, 0.0, 0.0],
        [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 78.7],
    ]
)


def isotropic_cij(K, G):
    lam = K - 2 * G / 3
    Cij = np.zeros((6, 6))
    Cij[:3, :3] = lam
    Cij[np.diag_indices(3)] = lam + 2 * G
    Cij[3, 3] = Cij[4, 4] = Cij[5, 5] = G
    return Cij


def voigt_moduli(Cij):
    """Voigt-average bulk and shear modulus of a stiffness matrix."""
    K = Cij[:3, :3].sum() / 9.0
    G = (np.trace(Cij[:3, :3]) - (Cij[0, 1] + Cij[0, 2] + Cij[1, 2])
         + 3 * (Cij[3, 3] + Cij[4, 4] + Cij[5, 5])) / 15.0
    return K, G


def make_odf(euler_angles, weights):
    euler_angles = np.atleast_2d(euler_angles)
    return pd.DataFrame(
        {
            "phi1": euler_angles[:, 0],
            "Phi": euler_angles[:, 1],
            "phi2": euler_angles[:, 2],
            "vol_fraction": weights,
        }
    )


def expect_error(fn, *args):
    try:
        fn(*args)
    except (ValueError, TypeError):
        return True
    return False


def test_single_phase_returns_input():
    tensors = OLIVINE[np.newaxis, :, :]
    fractions = np.array([1.0])
    assert np.allclose(voigt_volume_weighted_average(tensors, fractions), OLIVINE)
    assert np.allclose(reuss_volume_weighted_average(tensors, fractions), OLIVINE)


def test_two_isotropic_phases_closed_form():
    K1, G1, K2, G2 = 100.0, 60.0, 200.0, 90.0
    f1, f2 = 0.3, 0.7
    tensors = np.stack([isotropic_cij(K1, G1), isotropic_cij(K2, G2)])
    fractions = np.array([f1, f2])

    C_voigt = voigt_volume_weighted_average(tensors, fractions)
    C_reuss = reuss_volume_weighted_average(tensors, fractions)

    # Voigt = arithmetic mean, Reuss = harmonic mean of the moduli
    K_v, G_v = voigt_moduli(C_voigt)
    K_r, G_r = voigt_moduli(C_reuss)
    assert np.isclose(K_v, f1 * K1 + f2 * K2)
    assert np.isclose(G_v, f1 * G1 + f2 * G2)
    assert np.isclose(K_r, 1 / (f1 / K1 + f2 / K2))
    assert np.isclose(G_r, 1 / (f1 / G1 + f2 / G2))

    # classical bounds: Voigt stiffer than Reuss
    assert K_v >= K_r and G_v >= G_r


def test_fractions_are_normalised_with_warning():
    tensors = np.stack([isotropic_cij(100.0, 60.0), isotropic_cij(200.0, 90.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C = voigt_volume_weighted_average(tensors, np.array([30.0, 70.0]))
        assert any("normalising" in str(w.message) for w in caught)
    expected = voigt_volume_weighted_average(tensors, np.array([0.3, 0.7]))
    assert np.allclose(C, expected)


def test_cpo_identity_orientation_returns_input():
    odf = make_odf([[0.0, 0.0, 0.0]], [1.0])
    assert np.allclose(voigt_CPO_weighted_average(OLIVINE, odf), OLIVINE)
    assert np.allclose(reuss_CPO_weighted_average(OLIVINE, odf), OLIVINE)


def test_cpo_isotropic_tensor_is_invariant():
    iso = isotropic_cij(K=130.0, G=80.0)
    rng = np.random.default_rng(0)
    eulers = np.column_stack(
        [rng.uniform(0, 360, 25), rng.uniform(0, 180, 25), rng.uniform(0, 360, 25)]
    )
    odf = make_odf(eulers, np.full(25, 1.0 / 25))
    assert np.allclose(voigt_CPO_weighted_average(iso, odf), iso)
    assert np.allclose(reuss_CPO_weighted_average(iso, odf), iso)


def test_cpo_single_orientation_voigt_equals_reuss():
    # For one orientation both schemes reduce to "rotate the tensor",
    # so they must agree; rotation preserves the Voigt bulk modulus.
    odf = make_odf([[31.0, 58.0, 12.0]], [1.0])
    C_voigt = voigt_CPO_weighted_average(OLIVINE, odf)
    C_reuss = reuss_CPO_weighted_average(OLIVINE, odf)
    assert np.allclose(C_voigt, C_reuss, atol=1e-9)
    assert np.isclose(voigt_moduli(C_voigt)[0], voigt_moduli(OLIVINE)[0])


def test_rejects_bad_inputs():
    tensors = OLIVINE[np.newaxis, :, :]
    # shape errors
    assert expect_error(voigt_volume_weighted_average, OLIVINE, np.array([1.0]))
    assert expect_error(
        voigt_volume_weighted_average, tensors, np.array([0.5, 0.5])
    )
    # negative and all-zero fractions
    assert expect_error(
        reuss_volume_weighted_average,
        np.repeat(tensors, 2, axis=0), np.array([1.5, -0.5]),
    )
    assert expect_error(voigt_volume_weighted_average, tensors, np.array([0.0]))
    # CPO validation
    assert expect_error(voigt_CPO_weighted_average, OLIVINE, "not a dataframe")
    assert expect_error(
        voigt_CPO_weighted_average, OLIVINE,
        pd.DataFrame({"phi1": [0.0], "Phi": [0.0], "phi2": [0.0]}),
    )
    assert expect_error(
        reuss_CPO_weighted_average, np.zeros((3, 3)), make_odf([[0, 0, 0]], [1.0])
    )
    assert expect_error(
        reuss_CPO_weighted_average, OLIVINE, make_odf([[0, 0, 0]], [-1.0])
    )


def main():
    checks = [
        test_single_phase_returns_input,
        test_two_isotropic_phases_closed_form,
        test_fractions_are_normalised_with_warning,
        test_cpo_identity_orientation_returns_input,
        test_cpo_isotropic_tensor_is_invariant,
        test_cpo_single_orientation_voigt_equals_reuss,
        test_rejects_bad_inputs,
    ]
    ok = True
    for check in checks:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except AssertionError:
            ok = False
            print(f"[FAIL] {check.__name__}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
