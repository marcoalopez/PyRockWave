"""Validation of the Browaeys & Chevrot (2004) decomposition in decomposition.py.

The decomposition expresses a 6x6 elastic tensor as a sum of components, one per
symmetry class, obtained by projecting the 21-component elastic vector onto a
nested set of orthogonal subspaces (Browaeys & Chevrot 2004, eq. 3.4 and
Table 3). Rather than re-deriving the projector matrices, this test checks the
algebraic invariants the construction must satisfy regardless of how the
projectors are built:

  (1) Vector <-> tensor round-trip. ``_tensor_to_vector`` and
      ``_vector_to_tensor`` are exact inverses (eq. 2.2), so the composition is
      the identity for any symmetric tensor and any 21-vector.

  (2) Completeness. The cascade subtracts each projected component before the
      next projection, so the six components (including the triclinic
      "others" residual) must sum back to the original tensor exactly.

  (3) Parseval / orthogonality. Because the subspaces are mutually orthogonal in
      the metric induced by the sqrt(2)/2 weights of the elastic vector, the
      squared norm is additive: ||X||^2 == sum_class ||X_class||^2. This is the
      identity ``calc_percentages`` relies on, so the percentages of the six
      classes must sum to 100.

  (4) Idempotent projectors. Each ``_orthogonal_projector`` matrix M is a
      projection, hence M @ M == M.

  (5) Isotropic anchor. An analytically isotropic tensor (built from a bulk and
      shear modulus) must decompose to 100% isotropic, with the isotropic
      component reproducing the input and all lower-symmetry components ~0.

Plus the documented input-validation behaviour (ValueError on malformed input).

Run standalone (no pytest dependency):
    python tests/test_decomposition.py

References
----------
Browaeys, J.T., Chevrot, S., 2004. Decomposition of the elastic tensor and
    geophysical applications. Geophysical Journal International 159, 667-678.
    https://doi.org/10.1111/j.1365-246X.2004.02415.x
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pyrockwave.decomposition import (  # noqa: E402
    decompose_Cij,
    calc_percentages,
    _tensor_to_vector,
    _vector_to_tensor,
    _orthogonal_projector,
)


# --------------------------------------------------------------------------
# Test tensors
# --------------------------------------------------------------------------
SYMMETRY_CLASSES = (
    "isotropic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic",
)
ALL_KEYS = SYMMETRY_CLASSES + ("others",)

# San Carlos olivine (orthorhombic), GPa. A standard, fully triclinic-storage
# tensor that nonetheless has a strong isotropic part.
OLIVINE = np.array([
    [320.5,  68.1,  71.6,   0.0,   0.0,   0.0],
    [ 68.1, 196.5,  76.8,   0.0,   0.0,   0.0],
    [ 71.6,  76.8, 233.5,   0.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,  64.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,  77.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,  78.7],
])


def random_symmetric_tensor(seed=0):
    """A reproducible, fully populated symmetric 6x6 tensor (not necessarily
    physical) to exercise every component of the elastic vector."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6))
    return A + A.T


def isotropic_tensor(K, G):
    """Analytic isotropic stiffness tensor from bulk modulus K and shear
    modulus G (GPa): C11 = K + 4G/3, C12 = K - 2G/3, C44 = G."""
    c11 = K + 4.0 / 3.0 * G
    c12 = K - 2.0 / 3.0 * G
    c44 = G
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    np.fill_diagonal(C[:3, :3], c11)
    C[3, 3] = C[4, 4] = C[5, 5] = c44
    return C


# --------------------------------------------------------------------------
# Invariant checks (each returns a scalar residual / boolean)
# --------------------------------------------------------------------------
def roundtrip_tensor_error(Cij):
    """max|C - vector_to_tensor(tensor_to_vector(C))|."""
    return np.abs(Cij - _vector_to_tensor(_tensor_to_vector(Cij))).max()


def roundtrip_vector_error(seed=1):
    """max|X - tensor_to_vector(vector_to_tensor(X))| for a random 21-vector."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(21)
    return np.abs(X - _tensor_to_vector(_vector_to_tensor(X))).max()


def completeness_error(Cij):
    """max|C - sum of all decomposed components|."""
    parts = decompose_Cij(Cij)
    total = sum(parts[k] for k in ALL_KEYS)
    return np.abs(Cij - total).max()


def parseval_rel_error(Cij):
    """|  ||X||^2 - sum ||X_class||^2  | / ||X||^2."""
    parts = decompose_Cij(Cij)
    total_sq = np.linalg.norm(_tensor_to_vector(Cij)) ** 2
    sum_sq = sum(np.linalg.norm(_tensor_to_vector(parts[k])) ** 2 for k in ALL_KEYS)
    return abs(total_sq - sum_sq) / total_sq


def projector_idempotency_error():
    """max over classes of max|M @ M - M|."""
    worst = 0.0
    for cls in SYMMETRY_CLASSES:
        M = _orthogonal_projector(cls)
        worst = max(worst, np.abs(M @ M - M).max())
    return worst


def component_symmetry_error(Cij):
    """max over components of max|C_part - C_part.T|."""
    parts = decompose_Cij(Cij)
    return max(np.abs(parts[k] - parts[k].T).max() for k in ALL_KEYS)


def expect_value_error(fn, *args):
    """True iff calling fn(*args) raises ValueError."""
    try:
        fn(*args)
    except ValueError:
        return True
    return False


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    rand = random_symmetric_tensor()
    iso = isotropic_tensor(K=130.0, G=80.0)

    iso_parts = decompose_Cij(iso)
    iso_pct = calc_percentages(iso_parts)
    olivine_pct = calc_percentages(decompose_Cij(OLIVINE))

    print("=== invariant residuals ===")
    print(f"tensor round-trip (olivine):     {roundtrip_tensor_error(OLIVINE):.2e}")
    print(f"tensor round-trip (random):      {roundtrip_tensor_error(rand):.2e}")
    print(f"vector round-trip (random):      {roundtrip_vector_error():.2e}")
    print(f"completeness (olivine):          {completeness_error(OLIVINE):.2e}")
    print(f"completeness (random):           {completeness_error(rand):.2e}")
    print(f"Parseval rel. err (olivine):     {parseval_rel_error(OLIVINE):.2e}")
    print(f"Parseval rel. err (random):      {parseval_rel_error(rand):.2e}")
    print(f"projector idempotency:           {projector_idempotency_error():.2e}")
    print(f"component symmetry (random):     {component_symmetry_error(rand):.2e}")
    print(f"isotropic input -> isotropic %:  {iso_pct['isotropic']:.4f}")
    print(f"olivine isotropic %:             {olivine_pct['isotropic']:.2f}")
    print(f"olivine anisotropic %:           {olivine_pct['anisotropic']:.2f}")

    print("\n--- assertions ---")
    ok = True

    def check(name, condition):
        nonlocal ok
        ok = ok and bool(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    # (1) round-trips are exact to machine precision
    check("tensor round-trip exact (olivine)", roundtrip_tensor_error(OLIVINE) < 1e-10)
    check("tensor round-trip exact (random)", roundtrip_tensor_error(rand) < 1e-12)
    check("vector round-trip exact (random)", roundtrip_vector_error() < 1e-12)

    # (2) components reconstruct the original tensor
    check("completeness (olivine)", completeness_error(OLIVINE) < 1e-9)
    check("completeness (random)", completeness_error(rand) < 1e-12)

    # (3) squared-norm additivity (Parseval) -> percentages sum to 100
    check("Parseval additivity (olivine)", parseval_rel_error(OLIVINE) < 1e-9)
    check("Parseval additivity (random)", parseval_rel_error(rand) < 1e-9)
    class_sum = sum(olivine_pct[k] for k in ALL_KEYS)
    check("class percentages sum to 100 (olivine)", np.isclose(class_sum, 100.0, atol=1e-6))
    check("anisotropic == 100 - isotropic (olivine)",
          np.isclose(olivine_pct["anisotropic"], 100 - olivine_pct["isotropic"], atol=1e-9))

    # (4) projectors are idempotent
    check("projectors idempotent", projector_idempotency_error() < 1e-12)

    # components are symmetric tensors
    check("components symmetric (random)", component_symmetry_error(rand) < 1e-12)

    # (5) isotropic input decomposes to a single isotropic component
    check("isotropic input -> ~100% isotropic", np.isclose(iso_pct["isotropic"], 100.0, atol=1e-6))
    check("isotropic input -> ~0% anisotropic", iso_pct["anisotropic"] < 1e-6)
    check("isotropic component reproduces input",
          np.abs(iso_parts["isotropic"] - iso).max() < 1e-9)
    check("isotropic input -> ~0 lower-symmetry parts",
          max(np.abs(iso_parts[k]).max() for k in ("hexagonal", "tetragonal",
                                                    "orthorhombic", "monoclinic",
                                                    "others")) < 1e-9)

    # input validation
    check("decompose_Cij rejects wrong shape",
          expect_value_error(decompose_Cij, np.zeros((3, 3))))
    asymmetric = OLIVINE.copy()
    asymmetric[0, 1] += 5.0
    check("decompose_Cij rejects asymmetric tensor",
          expect_value_error(decompose_Cij, asymmetric))
    check("_tensor_to_vector rejects wrong shape",
          expect_value_error(_tensor_to_vector, np.zeros((3, 3))))
    check("_vector_to_tensor rejects wrong shape",
          expect_value_error(_vector_to_tensor, np.zeros(20)))
    check("_orthogonal_projector rejects unknown class",
          expect_value_error(_orthogonal_projector, "triclinic"))
    incomplete = {k: OLIVINE for k in SYMMETRY_CLASSES}  # missing "others"
    check("calc_percentages rejects missing keys",
          expect_value_error(calc_percentages, incomplete))

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


def test_decomposition_invariants():
    """Pytest entry point: the full standalone check suite must pass."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
