"""Tests for the Backus average in layered_media.py.

Checks the analytic anchors and invariants of the Backus (1962)
average of thin isotropic layers:

  (1) A uniform stack recovers the isotropic stiffness of its layers.
  (2) The result is a valid TI (VTI) tensor: symmetric, C11=C22,
      C44=C55, C13=C23, C66=(C11-C12)/2, zero off-block entries.
  (3) For two isotropic layers the result matches the Schoenberg &
      Muir calculus (independent implementation, exact for this case).
  (4) Volume fractions may be given as raw thicknesses (normalised
      internally); omitting them weights all layers equally.
  (5) The average is invariant to layer ordering.

Plus the documented input-validation behaviour (ValueError on
malformed input).

Run standalone (no pytest dependency):
    python tests/test_backus_average.py
"""

import sys

import numpy as np

from pyrockwave.layered_media import backus_average, schoenberg_muir_layered_medium


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def isotropic_cij(vp: float, vs: float, rho: float) -> np.ndarray:
    """6x6 isotropic stiffness (GPa) from velocities (km/s) and density (g/cm3)."""
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    m = lam + 2 * mu
    return np.array(
        [
            [m, lam, lam, 0, 0, 0],
            [lam, m, lam, 0, 0, 0],
            [lam, lam, m, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ]
    )


def random_stack(rng, n):
    """Random physically valid stack of n isotropic layers."""
    vp = rng.uniform(3.0, 7.0, n)
    vp_vs = rng.uniform(1.6, 2.1, n)  # > sqrt(4/3), typical rocks
    vs = vp / vp_vs
    rho = rng.uniform(2.0, 3.3, n)
    f = rng.uniform(0.1, 1.0, n)
    return vp, vs, rho, f


def test_uniform_stack_recovers_isotropic():
    vp, vs, rho = 6.0, 3.5, 2.7
    n = 5
    cij, rho_eff = backus_average(
        np.full(n, vp), np.full(n, vs), np.full(n, rho)
    )
    assert np.allclose(cij, isotropic_cij(vp, vs, rho))
    assert np.isclose(rho_eff, rho)


def test_ti_symmetry_invariants():
    rng = np.random.default_rng(0)
    for _ in range(20):
        vp, vs, rho, f = random_stack(rng, rng.integers(2, 40))
        cij, rho_eff = backus_average(vp, vs, rho, f)

        assert np.allclose(cij, cij.T)
        assert np.isclose(cij[0, 0], cij[1, 1])          # C11 = C22
        assert np.isclose(cij[3, 3], cij[4, 4])          # C44 = C55
        assert np.isclose(cij[0, 2], cij[1, 2])          # C13 = C23
        assert np.isclose(
            cij[5, 5], (cij[0, 0] - cij[0, 1]) / 2       # C66 = (C11-C12)/2
        )
        # off-block entries are exactly zero
        assert np.all(cij[:3, 3:] == 0) and np.all(cij[3:, :3] == 0)
        assert cij[3, 4] == cij[3, 5] == cij[4, 5] == 0
        # density is the volume-weighted mean
        assert np.isclose(rho_eff, np.sum(f / f.sum() * rho))


def test_matches_schoenberg_muir_two_layers():
    rng = np.random.default_rng(1)
    for _ in range(10):
        vp, vs, rho, f = random_stack(rng, 2)
        f = f / f.sum()

        cij_backus, _ = backus_average(vp, vs, rho, f)

        c1 = isotropic_cij(vp[0], vs[0], rho[0])
        c2 = isotropic_cij(vp[1], vs[1], rho[1])
        cij_sm, _, cij_sm_from_compliance = schoenberg_muir_layered_medium(
            c1, c2, float(f[0]), float(f[1])
        )

        assert np.allclose(cij_backus, cij_sm)
        assert np.allclose(cij_backus, cij_sm_from_compliance)


def test_fraction_normalisation_and_default():
    vp = np.array([4.0, 6.0])
    vs = np.array([2.2, 3.4])
    rho = np.array([2.3, 2.9])

    # raw thicknesses vs normalised fractions
    cij_a, rho_a = backus_average(vp, vs, rho, np.array([10.0, 30.0]))
    cij_b, rho_b = backus_average(vp, vs, rho, np.array([0.25, 0.75]))
    assert np.allclose(cij_a, cij_b) and np.isclose(rho_a, rho_b)

    # default weighting is equal fractions
    cij_c, rho_c = backus_average(vp, vs, rho)
    cij_d, rho_d = backus_average(vp, vs, rho, np.array([0.5, 0.5]))
    assert np.allclose(cij_c, cij_d) and np.isclose(rho_c, rho_d)


def test_order_invariance():
    rng = np.random.default_rng(2)
    vp, vs, rho, f = random_stack(rng, 15)
    cij, rho_eff = backus_average(vp, vs, rho, f)

    perm = rng.permutation(15)
    cij_p, rho_p = backus_average(vp[perm], vs[perm], rho[perm], f[perm])

    assert np.allclose(cij, cij_p)
    assert np.isclose(rho_eff, rho_p)


def test_rejects_bad_inputs():
    vp = np.array([4.0, 6.0])
    vs = np.array([2.2, 3.4])
    rho = np.array([2.3, 2.9])

    # mismatched lengths
    assert expect_value_error(backus_average, vp, vs[:1], rho)
    # empty input
    assert expect_value_error(
        backus_average, np.array([]), np.array([]), np.array([])
    )
    # non-positive velocity or density (fluid layers not supported)
    assert expect_value_error(backus_average, vp, np.array([0.0, 3.4]), rho)
    assert expect_value_error(backus_average, -vp, vs, rho)
    assert expect_value_error(backus_average, vp, vs, np.array([2.3, -2.9]))
    # NaN (common in well logs)
    assert expect_value_error(backus_average, np.array([4.0, np.nan]), vs, rho)
    # elastic stability: vp/vs must exceed sqrt(4/3)
    assert expect_value_error(
        backus_average, np.array([4.0, 3.5]), np.array([2.2, 3.4]), rho
    )
    # bad fractions: wrong length, zero, negative
    assert expect_value_error(backus_average, vp, vs, rho, np.array([1.0]))
    assert expect_value_error(backus_average, vp, vs, rho, np.array([0.0, 1.0]))
    assert expect_value_error(backus_average, vp, vs, rho, np.array([-0.5, 1.5]))


def main():
    checks = [
        test_uniform_stack_recovers_isotropic,
        test_ti_symmetry_invariants,
        test_matches_schoenberg_muir_two_layers,
        test_fraction_normalisation_and_default,
        test_order_invariance,
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
