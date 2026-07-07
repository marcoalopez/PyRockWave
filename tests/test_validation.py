"""Tests for the shared validators in utils/validation.py.

validate_cij must accept exactly the mechanically meaningful stiffness
matrices (symmetric, positive definite = Born stability) and reject
everything else. validate_wavevectors must accept (3,) and (n, 3)
arrays only.

Run standalone (no pytest dependency):
    python tests/test_validation.py
"""

import sys

import numpy as np

from pyrockwave.utils.validation import validate_cij, validate_wavevectors


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


def test_validate_cij_accepts_stable_tensor():
    assert validate_cij(isotropic_cij(K=130.0, G=80.0)) is True


def test_validate_cij_rejects_wrong_shape_or_type():
    assert expect_value_error(validate_cij, np.zeros((3, 3)))
    assert expect_value_error(validate_cij, isotropic_cij(130.0, 80.0).tolist())


def test_validate_cij_rejects_asymmetric():
    Cij = isotropic_cij(K=130.0, G=80.0)
    Cij[0, 1] += 5.0
    assert expect_value_error(validate_cij, Cij)


def test_validate_cij_rejects_unstable_tensor():
    # negative shear modulus -> not positive definite (Born criterion)
    assert expect_value_error(validate_cij, isotropic_cij(K=130.0, G=-10.0))
    assert expect_value_error(validate_cij, np.zeros((6, 6)))


def test_validate_wavevectors_accepts_valid_shapes():
    assert validate_wavevectors(np.array([1.0, 0.0, 0.0])) is True
    assert validate_wavevectors(np.zeros((10, 3))) is True


def test_validate_wavevectors_rejects_bad_shapes():
    assert expect_value_error(validate_wavevectors, [1.0, 0.0, 0.0])
    assert expect_value_error(validate_wavevectors, np.zeros(4))
    assert expect_value_error(validate_wavevectors, np.zeros((10, 4)))
    assert expect_value_error(validate_wavevectors, np.zeros((2, 3, 3)))


def main():
    checks = [
        test_validate_cij_accepts_stable_tensor,
        test_validate_cij_rejects_wrong_shape_or_type,
        test_validate_cij_rejects_asymmetric,
        test_validate_cij_rejects_unstable_tensor,
        test_validate_wavevectors_accepts_valid_shapes,
        test_validate_wavevectors_rejects_bad_shapes,
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
