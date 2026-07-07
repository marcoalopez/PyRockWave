"""Validation of the ElasticProps dataclass in elastic_tensor.py.

Checks the physical invariants the derived quantities must satisfy:

  (1) Isotropic anchor: for an analytically isotropic tensor built from
      a bulk modulus K and shear modulus G, the Voigt, Reuss and Hill
      averages all recover K and G exactly, both anisotropy indices are
      zero, and Poisson's ratio and the wave speeds match their
      closed-form expressions.
  (2) Bounds ordering: for an anisotropic crystal (olivine) the Voigt
      average is an upper bound and the Reuss average a lower bound,
      with Hill in between, for both K and G; the anisotropy indices
      are strictly positive.
  (3) Consistency: Sij is the exact inverse of Cij and the Browaeys &
      Chevrot percentages sum to 100.

Plus the documented input-validation behaviour and a smoke test of the
__repr__ summary.

Run standalone (no pytest dependency):
    python tests/test_elastic_tensor.py
"""

import sys

import numpy as np

from pyrockwave import ElasticProps

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


def make_props(Cij, density, **kwargs):
    return ElasticProps(
        temperature=25.0, pressure=1e-4, density=density, Cij=Cij, **kwargs
    )


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_isotropic_anchor():
    K, G, rho = 130.0, 80.0, 3.3
    props = make_props(isotropic_cij(K, G), rho)

    for value in (props.K_voigt, props.K_reuss, props.K_hill):
        assert np.isclose(value, K)
    for value in (props.G_voigt, props.G_reuss, props.G_hill):
        assert np.isclose(value, G)

    assert np.isclose(props.universal_anisotropy, 0.0)
    assert np.isclose(props.Kube_anisotropy, 0.0)

    nu = (3 * K - 2 * G) / (6 * K + 2 * G)
    assert np.isclose(props.isotropic_poisson_hill, nu)

    vp = np.sqrt((K + 4 * G / 3) / rho)
    vs = np.sqrt(G / rho)
    assert np.isclose(props.isotropic_vp_hill, vp, atol=1e-4)
    assert np.isclose(props.isotropic_vs_hill, vs, atol=1e-4)
    assert np.isclose(props.isotropic_vpvs_hill, vp / vs, atol=1e-4)

    assert np.isclose(props.percent["isotropic"], 100.0, atol=1e-6)


def test_voigt_reuss_hill_bounds_for_olivine():
    props = make_props(OLIVINE, RHO_OLIVINE, mineral_name="olivine",
                       crystal_system="orthorhombic")

    assert props.K_voigt > props.K_hill > props.K_reuss
    assert props.G_voigt > props.G_hill > props.G_reuss
    assert np.isclose(props.K_hill, (props.K_voigt + props.K_reuss) / 2)
    assert np.isclose(props.G_hill, (props.G_voigt + props.G_reuss) / 2)

    assert props.universal_anisotropy > 0
    assert props.Kube_anisotropy > 0

    # velocities ordered like the moduli they derive from
    assert props.isotropic_vp_voigt > props.isotropic_vp_reuss
    assert props.isotropic_vs_voigt > props.isotropic_vs_reuss


def test_internal_consistency():
    props = make_props(OLIVINE, RHO_OLIVINE)
    assert np.allclose(props.Sij @ props.Cij, np.eye(6), atol=1e-12)
    class_keys = ["isotropic", "hexagonal", "tetragonal",
                  "orthorhombic", "monoclinic", "others"]
    assert np.isclose(sum(props.percent[k] for k in class_keys), 100.0)
    assert set(props.decompose.keys()) == set(class_keys)


def test_rejects_bad_inputs():
    assert expect_value_error(make_props, np.zeros((3, 3)), 3.3)
    assert expect_value_error(make_props, OLIVINE.tolist(), 3.3)
    assert expect_value_error(make_props, OLIVINE, -1.0)
    asymmetric = OLIVINE.copy()
    asymmetric[0, 1] += 5.0
    assert expect_value_error(make_props, asymmetric, 3.3)
    assert expect_value_error(make_props, OLIVINE, 3.3, crystal_system="pentagonal")
    # crystal system is case-insensitive: this must NOT raise
    make_props(OLIVINE, 3.3, crystal_system="Orthorhombic")


def test_repr_smoke():
    props = make_props(OLIVINE, RHO_OLIVINE, mineral_name="olivine",
                       crystal_system="orthorhombic")
    text = repr(props)
    assert "olivine" in text
    assert "Elastic Tensor" in text


def main():
    checks = [
        test_isotropic_anchor,
        test_voigt_reuss_hill_bounds_for_olivine,
        test_internal_consistency,
        test_rejects_bad_inputs,
        test_repr_smoke,
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
