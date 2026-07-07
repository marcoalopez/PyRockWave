"""Validation of the analytic anisotropy models in anisotropic_models.py.

The VTI (polar) models have closed-form structure that can be checked
without stored reference values:

  (1) Thomsen parameter round trip: a VTI tensor built from prescribed
      (Vp0, Vs0, epsilon, delta, gamma) must return exactly those
      parameters through Thomsen_params.
  (2) Isotropic limit: for an isotropic tensor, epsilon = delta =
      gamma = 0 and both the weak (Thomsen 1986) and exact
      (Anderson 1961) models predict direction-independent velocities
      with zero shear-wave splitting.
  (3) Exact model vs Christoffel: Anderson's closed-form VTI solution
      must match the numerical Christoffel eigenvalue solution of
      christoffel.py for the same tensor to machine precision.
  (4) Weak vs exact: for weak anisotropy the Thomsen linearisation must
      approach the exact solution (small relative error).
  (5) Vertical propagation: all models recover Vp0 at polar angle 0.
  (6) Orthotropic (Hao & Stovas / Wang et al.) model: exact in the
      isotropic limit and along the three coordinate axes (where
      Vp = sqrt(c33/rho), sqrt(c11/rho), sqrt(c22/rho) for any
      anisotropy strength, since r2*eps2^2 = c11/c33 and
      r1*eps1^2 = c22/c33 identically), azimuth-independent for VTI
      input, and close to the exact Christoffel solution for weak
      orthorhombic anisotropy.

Run standalone (no pytest dependency):
    python tests/test_anisotropic_models.py
"""

import sys

import numpy as np

from pyrockwave.anisotropic_models import (
    weak_polar_anisotropy,
    polar_anisotropy,
    orthotropic_azimuthal_anisotropy,
    Thomsen_params,
)
from pyrockwave.christoffel import phase_seismic_properties


def isotropic_cij(K, G):
    lam = K - 2 * G / 3
    Cij = np.zeros((6, 6))
    Cij[:3, :3] = lam
    Cij[np.diag_indices(3)] = lam + 2 * G
    Cij[3, 3] = Cij[4, 4] = Cij[5, 5] = G
    return Cij


def vti_tensor(Vp0, Vs0, epsilon, delta, gamma, rho):
    """Build a VTI stiffness tensor (GPa) from Thomsen parameters."""
    c33 = rho * Vp0**2
    c44 = rho * Vs0**2
    c11 = c33 * (1 + 2 * epsilon)
    c66 = c44 * (1 + 2 * gamma)
    c13 = np.sqrt(2 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2) - c44
    c12 = c11 - 2 * c66
    Cij = np.array(
        [
            [c11, c12, c13, 0.0, 0.0, 0.0],
            [c12, c11, c13, 0.0, 0.0, 0.0],
            [c13, c13, c33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c66],
        ]
    )
    return Cij


def sample_directions(n=40):
    rng = np.random.default_rng(0)
    azimuths = rng.uniform(0, 2 * np.pi, n)
    polar = rng.uniform(0, np.pi / 2, n)
    return azimuths, polar


def test_thomsen_params_roundtrip():
    Vp0, Vs0, eps, delta, gamma, rho = 6.0, 3.5, 0.10, 0.05, 0.08, 2.7
    Cij = vti_tensor(Vp0, Vs0, eps, delta, gamma, rho)
    Vp0_out, Vs0_out, eps_out, delta_out, gamma_out = Thomsen_params(Cij, rho)
    assert np.isclose(Vp0_out, Vp0)
    assert np.isclose(Vs0_out, Vs0)
    assert np.isclose(eps_out, eps)
    assert np.isclose(delta_out, delta)
    assert np.isclose(gamma_out, gamma)


def test_isotropic_limit():
    K, G, rho = 130.0, 80.0, 3.3
    Cij = isotropic_cij(K, G)
    vp = np.sqrt((K + 4 * G / 3) / rho)
    vs = np.sqrt(G / rho)

    _, _, eps, delta, gamma = Thomsen_params(Cij, rho)
    assert np.isclose(eps, 0) and np.isclose(delta, 0) and np.isclose(gamma, 0)

    wavevectors = sample_directions()
    for model in (weak_polar_anisotropy, polar_anisotropy):
        df = model(Cij, rho, wavevectors)
        assert np.allclose(df["Vp"], vp)
        assert np.allclose(df["Vsv"], vs)
        assert np.allclose(df["Vsh"], vs)
        assert np.allclose(df["SWS"], 0.0)


def test_exact_polar_model_matches_christoffel():
    rho = 2.7
    Cij = vti_tensor(6.0, 3.5, 0.10, 0.05, 0.08, rho)
    azimuths, polar = sample_directions()

    df_model = polar_anisotropy(Cij, rho, (azimuths, polar))
    df_chris, _ = phase_seismic_properties(
        Cij, rho, np.rad2deg(azimuths), np.rad2deg(polar)
    )

    assert np.allclose(df_model["Vp"], df_chris["Vp_phase_kms"], rtol=1e-8)
    assert np.allclose(df_model["Vs1"], df_chris["Vs1_phase_kms"], rtol=1e-8)
    assert np.allclose(df_model["Vs2"], df_chris["Vs2_phase_kms"], rtol=1e-8)


def test_weak_model_approaches_exact_for_weak_anisotropy():
    rho = 2.7
    Cij = vti_tensor(6.0, 3.5, 0.03, 0.02, 0.02, rho)
    wavevectors = sample_directions()

    df_weak = weak_polar_anisotropy(Cij, rho, wavevectors)
    df_exact = polar_anisotropy(Cij, rho, wavevectors)

    for col in ("Vp", "Vsv", "Vsh"):
        assert np.allclose(df_weak[col], df_exact[col], rtol=5e-3)


def test_vertical_propagation_recovers_Vp0():
    rho = 2.7
    Vp0, Vs0 = 6.0, 3.5
    Cij = vti_tensor(Vp0, Vs0, 0.10, 0.05, 0.08, rho)
    vertical = (np.array([0.0]), np.array([0.0]))

    for model in (weak_polar_anisotropy, polar_anisotropy):
        df = model(Cij, rho, vertical)
        assert np.isclose(df["Vp"][0], Vp0)
        assert np.isclose(df["Vsv"][0], Vs0)
        assert np.isclose(df["Vsh"][0], Vs0)

    df = orthotropic_azimuthal_anisotropy(Cij, rho, vertical)
    assert np.isclose(df["Vp"][0], Vp0)


def test_orthotropic_model_isotropic_limit():
    K, G, rho = 130.0, 80.0, 3.3
    vp = np.sqrt((K + 4 * G / 3) / rho)
    df = orthotropic_azimuthal_anisotropy(isotropic_cij(K, G), rho, sample_directions())
    assert np.allclose(df["Vp"], vp)


def test_orthotropic_model_vti_is_azimuth_independent():
    # a VTI medium is a special orthorhombic medium with no azimuthal
    # dependence: fixed polar angle, sweeping azimuth must give one Vp
    rho = 2.7
    Cij = vti_tensor(6.0, 3.5, 0.10, 0.05, 0.08, rho)
    azimuths = np.linspace(0, 2 * np.pi, 37)
    polar = np.full_like(azimuths, np.deg2rad(45.0))
    df = orthotropic_azimuthal_anisotropy(Cij, rho, (azimuths, polar))
    assert np.allclose(df["Vp"], df["Vp"][0])


def test_orthotropic_model_axis_anchors():
    # along the coordinate axes the model is exact for any anisotropy
    # strength: Vp(vertical) = sqrt(c33/rho), Vp(x) = sqrt(c11/rho),
    # Vp(y) = sqrt(c22/rho). Olivine, strongly anisotropic.
    Cij = np.array(
        [
            [320.5, 68.15, 71.6, 0.0, 0.0, 0.0],
            [68.15, 196.5, 76.8, 0.0, 0.0, 0.0],
            [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 78.7],
        ]
    )
    rho = 3.355
    azimuths = np.array([0.0, 0.0, np.pi / 2])
    polar = np.array([0.0, np.pi / 2, np.pi / 2])
    df = orthotropic_azimuthal_anisotropy(Cij, rho, (azimuths, polar))
    assert np.isclose(df["Vp"][0], np.sqrt(Cij[2, 2] / rho))
    assert np.isclose(df["Vp"][1], np.sqrt(Cij[0, 0] / rho))
    assert np.isclose(df["Vp"][2], np.sqrt(Cij[1, 1] / rho))


def test_orthotropic_model_approaches_christoffel_for_weak_anisotropy():
    # weakly orthorhombic medium: the approximate model should track the
    # exact Christoffel P velocity to within the linearisation error
    rho = 2.7
    Cij = isotropic_cij(K=60.0, G=30.0)
    Cij[0, 0] *= 1.06
    Cij[1, 1] *= 1.03
    Cij[5, 5] *= 1.02
    azimuths, polar = sample_directions()
    df_model = orthotropic_azimuthal_anisotropy(Cij, rho, (azimuths, polar))
    df_chris, _ = phase_seismic_properties(
        Cij, rho, np.rad2deg(azimuths), np.rad2deg(polar)
    )
    assert np.allclose(df_model["Vp"], df_chris["Vp_phase_kms"], rtol=1e-2)


def main():
    checks = [
        test_thomsen_params_roundtrip,
        test_isotropic_limit,
        test_exact_polar_model_matches_christoffel,
        test_weak_model_approaches_exact_for_weak_anisotropy,
        test_vertical_propagation_recovers_Vp0,
        test_orthotropic_model_isotropic_limit,
        test_orthotropic_model_vti_is_azimuth_independent,
        test_orthotropic_model_axis_anchors,
        test_orthotropic_model_approaches_christoffel_for_weak_anisotropy,
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
