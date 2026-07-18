"""Validation of the Christoffel solver in christoffel.py.

Anchored on analytic solutions rather than stored reference numbers:

  (1) Cubic crystal (GaAs, same constants as the repository notebook
      test_christoffel_GaAs.ipynb). For cubic symmetry the Christoffel
      equation has closed-form solutions along the high-symmetry
      directions:
        [001]: Vp = sqrt(c11/rho),              Vs1 = Vs2 = sqrt(c44/rho)
        [110]: Vp = sqrt((c11+c12+2c44)/2rho),  Vs = sqrt(c44/rho) and
                                                     sqrt((c11-c12)/2rho)
        [111]: Vp = sqrt((c11+2c12+4c44)/3rho), Vs1 = Vs2 =
                                                     sqrt((c11-c12+c44)/3rho)
  (2) Isotropic medium: velocities are direction-independent and equal
      to sqrt((K+4/3G)/rho) and sqrt(G/rho); group velocity equals phase
      velocity; power flow angles are zero; the P-wave enhancement
      factor is 1 (S modes are degenerate everywhere, where enhancement
      is documented as unreliable, so they are not checked).
  (3) Polarizations: the eigenvector triad is orthonormal for every
      direction, and the P polarization is parallel to the propagation
      direction in an isotropic medium.
  (4) phase_seismic_properties and full_seismic_properties agree on
      their shared phase-velocity columns.
  (5) Cartesian wavevector input is equivalent to the spherical-angle
      input (identical outputs), scale-invariant (vectors normalised
      internally), and accepts a single (3,) direction.

Plus the documented input-validation behaviour.

Run standalone (no pytest dependency):
    python tests/test_christoffel.py
"""

import sys

import numpy as np

from pyrockwave.christoffel import phase_seismic_properties, full_seismic_properties

# GaAs, cubic (as in notebooks/test_christoffel_GaAs.ipynb), GPa and g/cm3
C11, C12, C44 = 118.8, 53.8, 59.4
RHO_GAAS = 5.307
GAAS = np.array(
    [
        [C11, C12, C12, 0.0, 0.0, 0.0],
        [C12, C11, C12, 0.0, 0.0, 0.0],
        [C12, C12, C11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C44],
    ]
)

# Olivine at ambient conditions (Abramson et al., 1997), GPa and g/cm3
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
RHO_OLIVINE = 3.355


def isotropic_cij(K, G):
    lam = K - 2 * G / 3
    Cij = np.zeros((6, 6))
    Cij[:3, :3] = lam
    Cij[np.diag_indices(3)] = lam + 2 * G
    Cij[3, 3] = Cij[4, 4] = Cij[5, 5] = G
    return Cij


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_cubic_high_symmetry_directions():
    # [001], [110], [111] as (azimuth, polar) in degrees
    polar_111 = np.rad2deg(np.arccos(1 / np.sqrt(3)))
    azimuths = np.array([0.0, 45.0, 45.0])
    polar = np.array([0.0, 90.0, polar_111])

    df, _ = phase_seismic_properties(GAAS, RHO_GAAS, azimuths, polar)

    # [001]
    assert np.isclose(df["Vp_phase_kms"][0], np.sqrt(C11 / RHO_GAAS))
    assert np.isclose(df["Vs1_phase_kms"][0], np.sqrt(C44 / RHO_GAAS))
    assert np.isclose(df["Vs2_phase_kms"][0], np.sqrt(C44 / RHO_GAAS))

    # [110]
    assert np.isclose(df["Vp_phase_kms"][1], np.sqrt((C11 + C12 + 2 * C44) / (2 * RHO_GAAS)))
    shear_moduli = sorted([C44, (C11 - C12) / 2])
    assert np.isclose(df["Vs2_phase_kms"][1], np.sqrt(shear_moduli[0] / RHO_GAAS))
    assert np.isclose(df["Vs1_phase_kms"][1], np.sqrt(shear_moduli[1] / RHO_GAAS))

    # [111]
    assert np.isclose(df["Vp_phase_kms"][2], np.sqrt((C11 + 2 * C12 + 4 * C44) / (3 * RHO_GAAS)))
    vs_111 = np.sqrt((C11 - C12 + C44) / (3 * RHO_GAAS))
    assert np.isclose(df["Vs1_phase_kms"][2], vs_111)
    assert np.isclose(df["Vs2_phase_kms"][2], vs_111)


def test_isotropic_medium_velocities_and_group():
    K, G, rho = 130.0, 80.0, 3.3
    vp = np.sqrt((K + 4 * G / 3) / rho)
    vs = np.sqrt(G / rho)

    rng = np.random.default_rng(0)
    azimuths = rng.uniform(0, 360, 50)
    polar = rng.uniform(0, 180, 50)

    df, _ = full_seismic_properties(isotropic_cij(K, G), rho, azimuths, polar)

    assert np.allclose(df["Vp_phase_kms"], vp)
    assert np.allclose(df["Vs1_phase_kms"], vs)
    assert np.allclose(df["Vs2_phase_kms"], vs)

    # group velocity == phase velocity, energy travels along the ray
    assert np.allclose(df["Vp_group_kms"], vp)
    assert np.allclose(df["Vs1_group_kms"], vs)
    assert np.allclose(df["Vs2_group_kms"], vs)
    # atol in degrees: arccos amplifies float noise near cos = 1, so
    # angles up to ~1e-6 deg are numerically indistinguishable from zero
    assert np.allclose(df["Vp_powerflow_deg"], 0.0, atol=1e-4)
    assert np.allclose(df["Vs1_powerflow_deg"], 0.0, atol=1e-4)
    assert np.allclose(df["Vs2_powerflow_deg"], 0.0, atol=1e-4)

    # no focusing/defocusing of P energy in an isotropic medium
    assert np.allclose(df["Vp_enhancement"], 1.0)


def test_polarizations_orthonormal_and_isotropic_P_longitudinal():
    rng = np.random.default_rng(1)
    azimuths = rng.uniform(0, 360, 40)
    polar = rng.uniform(0, 180, 40)

    # olivine: orthonormal triads for an anisotropic crystal
    df, eigenvectors = phase_seismic_properties(OLIVINE, RHO_OLIVINE, azimuths, polar)
    gram = np.einsum("nmi,nki->nmk", eigenvectors, eigenvectors)
    assert np.allclose(gram, np.eye(3), atol=1e-10)

    # isotropic: P polarization (mode index 2) parallel to propagation
    df, eigenvectors = phase_seismic_properties(
        isotropic_cij(130.0, 80.0), 3.3, azimuths, polar
    )
    q = df[["qx", "qy", "qz"]].to_numpy()
    cos_pq = np.abs(np.einsum("ni,ni->n", eigenvectors[:, 2, :], q))
    assert np.allclose(cos_pq, 1.0)


def test_phase_and_full_agree_on_shared_columns():
    azimuths = np.linspace(0, 350, 20)
    polar = np.linspace(5, 175, 20)
    df_phase, _ = phase_seismic_properties(OLIVINE, RHO_OLIVINE, azimuths, polar)
    df_full, _ = full_seismic_properties(OLIVINE, RHO_OLIVINE, azimuths, polar)
    for col in ["qx", "qy", "qz", "Vs2_phase_kms", "Vs1_phase_kms", "Vp_phase_kms"]:
        assert np.allclose(df_phase[col], df_full[col])


def test_cartesian_input_matches_spherical():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(30, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    azimuths = np.rad2deg(np.arctan2(q[:, 1], q[:, 0]) % (2 * np.pi))
    polar = np.rad2deg(np.arccos(q[:, 2]))

    for fn in (phase_seismic_properties, full_seismic_properties):
        df_sph, eig_sph = fn(OLIVINE, RHO_OLIVINE, azimuths, polar)
        df_cart, eig_cart = fn(OLIVINE, RHO_OLIVINE, wavevectors=q)
        assert np.allclose(df_sph.to_numpy(), df_cart.to_numpy())
        assert np.allclose(eig_sph, eig_cart)

    # non-unit vectors are normalised internally
    df_unit, _ = phase_seismic_properties(OLIVINE, RHO_OLIVINE, wavevectors=q)
    df_scaled, _ = phase_seismic_properties(OLIVINE, RHO_OLIVINE, wavevectors=3.7 * q)
    assert np.allclose(df_unit.to_numpy(), df_scaled.to_numpy())

    # a single direction can be passed as shape (3,)
    df_one, _ = phase_seismic_properties(
        OLIVINE, RHO_OLIVINE, wavevectors=np.array([0.0, 0.0, 1.0])
    )
    assert len(df_one) == 1
    assert np.isclose(df_one["polar_deg"][0], 0.0)


def test_rejects_bad_inputs():
    az = np.array([0.0, 45.0])
    po = np.array([90.0, 90.0])
    q = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    # mismatched shapes
    assert expect_value_error(
        phase_seismic_properties, OLIVINE, RHO_OLIVINE, az, po[:1]
    )
    # not 1D
    assert expect_value_error(
        phase_seismic_properties, OLIVINE, RHO_OLIVINE, az.reshape(2, 1), po.reshape(2, 1)
    )
    # non-positive density
    assert expect_value_error(phase_seismic_properties, OLIVINE, 0.0, az, po)
    # mechanically unstable tensor
    assert expect_value_error(phase_seismic_properties, np.eye(6) * -1, 3.3, az, po)
    # directions in both forms, or in neither, or half a spherical pair
    assert expect_value_error(
        lambda: phase_seismic_properties(OLIVINE, RHO_OLIVINE, az, po, q)
    )
    assert expect_value_error(
        lambda: phase_seismic_properties(OLIVINE, RHO_OLIVINE)
    )
    assert expect_value_error(
        lambda: phase_seismic_properties(OLIVINE, RHO_OLIVINE, azimuths_deg=az)
    )
    # malformed wavevectors: wrong shape, zero-norm, non-array
    assert expect_value_error(
        lambda: phase_seismic_properties(
            OLIVINE, RHO_OLIVINE, wavevectors=np.zeros((2, 2))
        )
    )
    assert expect_value_error(
        lambda: phase_seismic_properties(
            OLIVINE, RHO_OLIVINE, wavevectors=np.array([[0.0, 0.0, 0.0]])
        )
    )
    assert expect_value_error(
        lambda: phase_seismic_properties(
            OLIVINE, RHO_OLIVINE, wavevectors=[[0.0, 0.0, 1.0]]
        )
    )


def main():
    checks = [
        test_cubic_high_symmetry_directions,
        test_isotropic_medium_velocities_and_group,
        test_polarizations_orthonormal_and_isotropic_P_longitudinal,
        test_phase_and_full_agree_on_shared_columns,
        test_cartesian_input_matches_spherical,
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
