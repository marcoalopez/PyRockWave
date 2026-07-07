"""Tests for the tensor utilities in utils/tensor_tools.py.

Checks the algebraic invariants that stiffness-tensor rotation must
satisfy regardless of implementation details:

  (1) Voigt <-> 4th-rank round trip: _rearrange_tensor and
      _tensor_in_voigt are exact inverses for any symmetric 6x6 matrix.
  (2) Identity rotations: rotating by 0 deg or 360 deg leaves the tensor
      unchanged; rotating by theta then -theta restores the original.
  (3) Symmetry preservation: a rotated stiffness matrix stays symmetric.
  (4) Isotropic invariance: an isotropic tensor is unchanged by any
      rotation about any axis.
  (5) Cubic symmetry: a cubic tensor is invariant under 90 deg rotations
      about the crystal axes.
  (6) Invariants: the Voigt bulk modulus (a rotational invariant of the
      stiffness tensor) is preserved by arbitrary rotations.

Plus the documented input-validation behaviour.

Run standalone (no pytest dependency):
    python tests/test_tensor_tools.py
"""

import sys

import numpy as np

from pyrockwave.utils.tensor_tools import (
    rotate_stiffness_tensor,
    _rearrange_tensor,
    _tensor_in_voigt,
    _normalize_vector,
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


def cubic_cij(c11, c12, c44):
    Cij = np.zeros((6, 6))
    Cij[:3, :3] = c12
    Cij[np.diag_indices(3)] = c11
    Cij[3, 3] = Cij[4, 4] = Cij[5, 5] = c44
    return Cij


def voigt_bulk_modulus(Cij):
    return (Cij[:3, :3].sum()) / 9.0


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_voigt_tensor_roundtrip():
    Cijkl = _rearrange_tensor(OLIVINE)
    assert Cijkl.shape == (3, 3, 3, 3)
    assert np.allclose(_tensor_in_voigt(Cijkl), OLIVINE)


def test_identity_rotations():
    for axis in ("x", "y", "z"):
        rotated, _ = rotate_stiffness_tensor(OLIVINE, 0.0, axis)
        assert np.allclose(rotated, OLIVINE)
        rotated, _ = rotate_stiffness_tensor(OLIVINE, 360.0, axis)
        assert np.allclose(rotated, OLIVINE)


def test_forward_backward_rotation():
    rotated, _ = rotate_stiffness_tensor(OLIVINE, 37.0, "y")
    restored, _ = rotate_stiffness_tensor(rotated, -37.0, "y")
    assert np.allclose(restored, OLIVINE)


def test_rotation_preserves_symmetry():
    rotated, _ = rotate_stiffness_tensor(OLIVINE, 23.0, "x")
    assert np.allclose(rotated, rotated.T)


def test_isotropic_tensor_is_rotation_invariant():
    iso = isotropic_cij(K=130.0, G=80.0)
    for axis, angle in (("x", 17.0), ("y", 45.0), ("z", 71.3)):
        rotated, _ = rotate_stiffness_tensor(iso, angle, axis)
        assert np.allclose(rotated, iso)


def test_cubic_tensor_invariant_under_90deg_rotations():
    cubic = cubic_cij(c11=118.8, c12=53.8, c44=59.4)  # GaAs
    for axis in ("x", "y", "z"):
        rotated, _ = rotate_stiffness_tensor(cubic, 90.0, axis)
        assert np.allclose(rotated, cubic)


def test_rotation_preserves_voigt_bulk_modulus():
    K_before = voigt_bulk_modulus(OLIVINE)
    rotated, _ = rotate_stiffness_tensor(OLIVINE, 52.0, "z")
    rotated, _ = rotate_stiffness_tensor(rotated, 31.0, "x")
    assert np.isclose(voigt_bulk_modulus(rotated), K_before)


def test_rotate_rejects_bad_inputs():
    assert expect_value_error(rotate_stiffness_tensor, [[1.0]], 10.0)
    assert expect_value_error(rotate_stiffness_tensor, np.zeros((5, 5)), 10.0)
    assert expect_value_error(rotate_stiffness_tensor, OLIVINE, 10.0, "w")
    assert expect_value_error(rotate_stiffness_tensor, OLIVINE, 10.0, 3)


def test_normalize_vector():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(np.linalg.norm(_normalize_vector(v)), 1.0)
    zero = np.zeros(3)
    assert np.allclose(_normalize_vector(zero), zero)
    assert expect_value_error(_normalize_vector, np.zeros(4))


def main():
    checks = [
        test_voigt_tensor_roundtrip,
        test_identity_rotations,
        test_forward_backward_rotation,
        test_rotation_preserves_symmetry,
        test_isotropic_tensor_is_rotation_invariant,
        test_cubic_tensor_invariant_under_90deg_rotations,
        test_rotation_preserves_voigt_bulk_modulus,
        test_rotate_rejects_bad_inputs,
        test_normalize_vector,
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
