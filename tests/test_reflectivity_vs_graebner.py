"""Independent validation of layered_media.reflectivity().

Cross-checks the module's Ruger (1998) symmetry-plane PP reflectivity against
the *exact* VTI plane-wave PP reflection coefficient (Graebner 1992; Daley &
Hron 1979), computed here from first principles by solving the Christoffel
equation for the qP/qSV vertical slownesses and the P-SV boundary-value
problem. The two implementations share no code or derivation path.

For an interface between two VTI media the orthorhombic symmetry-plane
reflectivities Rxz and Ryz both reduce to the VTI PP response, so they can be
compared directly with the exact solution. Because the module is a weak-
contrast / weak-anisotropy linearisation, it is expected to converge to the
exact result as contrast and anisotropy decrease, and to deviate increasingly
(smoothly, by design) for stronger contrast and larger angles.

Two convention-independent anchors validate the exact solver itself:
  (A) at normal incidence it must equal the acoustic-impedance reflection
      (Z2 - Z1) / (Z2 + Z1);
  (B) in the isotropic limit it must match the module's Aki-Richards form.

Run standalone (no pytest dependency):
    python tests/test_reflectivity_vs_graebner.py

References
----------
Graebner, M., 1992, Plane-wave reflection and transmission coefficients for a
    transversely isotropic solid: Geophysics, 57, 1512-1519.
Ruger, A., 1998, Variation of P-wave reflectivity with offset and azimuth in
    anisotropic media: Geophysics, 63, 935-947.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pyrockwave.layered_media import tsvankin_params, reflectivity  # noqa: E402


# --------------------------------------------------------------------------
# Exact VTI P-SV reflection coefficient (Graebner 1992)
# --------------------------------------------------------------------------
def _q_roots(c11, c33, c44, c13, rho, p):
    """Vertical slownesses qP, qSV (> 0) for horizontal slowness p, from the
    P-SV Christoffel determinant A q^4 + B q^2 + C = 0."""
    A = c33 * c44
    B = (c11 * c33 + c44**2 - (c13 + c44)**2) * p**2 - rho * (c44 + c33)
    C = c11 * c44 * p**4 - rho * (c11 + c44) * p**2 + rho**2
    sq = np.sqrt(B**2 - 4 * A * C)
    q2_small = (-B - sq) / (2 * A)   # qP  (faster wave -> smaller slowness)
    q2_large = (-B + sq) / (2 * A)   # qSV
    return np.sqrt(q2_small), np.sqrt(q2_large)


def _polarization(c11, c33, c44, c13, rho, p, q, mode):
    """Unit polarization (Ux, Uz) from the Christoffel null space. For P modes
    the vector is oriented along the slowness direction (p, q)."""
    M11 = c11 * p**2 + c44 * q**2 - rho
    M12 = (c13 + c44) * p * q
    M22 = c44 * p**2 + c33 * q**2 - rho
    v1 = np.array([M12, -M11])
    v2 = np.array([M22, -M12])
    v = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
    v = v / np.linalg.norm(v)
    if mode == "P" and np.dot(v, np.array([p, q])) < 0:
        v = -v
    return v


def _b_vector(c13, c33, c44, p, q, U):
    """Displacement-stress vector [ux, uz, sigma_zz, sigma_xz]; the common
    factor i*omega is dropped (it cancels in the boundary-condition system)."""
    Ux, Uz = U
    szz = c13 * p * Ux + c33 * q * Uz
    sxz = c44 * (q * Ux + p * Uz)
    return np.array([Ux, Uz, szz, sxz])


def vti_exact_rpp(m_upper, m_lower, p):
    """Exact PP reflection coefficient for an incident P-wave of horizontal
    slowness p. Each medium tuple is (c11, c33, c44, c13, rho).

    Returns (Rpp, qP_upper) where qP_upper is the incident vertical slowness,
    used to recover the incidence phase angle theta = atan2(p, qP_upper).
    """
    qP1, qSV1 = _q_roots(*m_upper, p)
    qP2, qSV2 = _q_roots(*m_lower, p)

    def bvec(m, p, q, mode):
        c11, c33, c44, c13, rho = m
        U = _polarization(c11, c33, c44, c13, rho, p, q, mode)
        return _b_vector(c13, c33, c44, p, q, U)

    b_inc = bvec(m_upper, p, +qP1, "P")     # incident P (downgoing)
    b_rP = bvec(m_upper, p, -qP1, "P")      # reflected P (upgoing)
    b_rSV = bvec(m_upper, p, -qSV1, "SV")   # reflected SV (upgoing)
    b_tP = bvec(m_lower, p, +qP2, "P")      # transmitted P (downgoing)
    b_tSV = bvec(m_lower, p, +qSV2, "SV")   # transmitted SV (downgoing)

    system = np.column_stack([b_rP, b_rSV, -b_tP, -b_tSV])
    coeffs = np.linalg.solve(system, -b_inc)
    return coeffs[0], qP1


# --------------------------------------------------------------------------
# Build a VTI stiffness tensor from velocities and Thomsen parameters
# --------------------------------------------------------------------------
def vti_tensor(Vp0, Vs0, eps, delta, rho, gamma=0.0):
    """Return (6x6 stiffness in GPa, (c11, c33, c44, c13, rho)) for a VTI
    medium given vertical velocities (km/s), Thomsen eps/delta/gamma and
    density (g/cm^3)."""
    c33 = rho * Vp0**2
    c44 = rho * Vs0**2
    c11 = c33 * (1 + 2 * eps)
    c13 = -c44 + np.sqrt(2 * c33 * (c33 - c44) * delta + (c33 - c44)**2)
    c66 = c44 * (1 + 2 * gamma)
    c12 = c11 - 2 * c66
    C = np.array([
        [c11, c12, c13,   0,   0,   0],
        [c12, c11, c13,   0,   0,   0],
        [c13, c13, c33,   0,   0,   0],
        [  0,   0,   0, c44,   0,   0],
        [  0,   0,   0,   0, c44,   0],
        [  0,   0,   0,   0,   0, c66],
    ])
    return C, (c11, c33, c44, c13, rho)


def compare(upper, lower, theta_max_deg=35.0, n=15):
    """Compare module reflectivity with the exact VTI solution across angle.

    Returns a dict of diagnostics. `upper`/`lower` are dicts of Vp0, Vs0,
    eps, delta, rho (km/s, g/cm^3)."""
    C_up, m_up = vti_tensor(**upper)
    C_low, m_low = vti_tensor(**lower)
    rho_up, rho_low = upper["rho"], lower["rho"]

    # Sweep horizontal slowness; recover the incidence phase angle for each.
    vp0_up = np.sqrt(m_up[1] / rho_up)
    p_max = np.sin(np.deg2rad(theta_max_deg)) / vp0_up
    thetas, r_exact = [], []
    for p in np.linspace(1e-6, p_max, n):
        rpp, qP1 = vti_exact_rpp(m_up, m_low, p)
        thetas.append(np.arctan2(p, qP1))
        r_exact.append(rpp)
    thetas = np.array(thetas)
    r_exact = np.array(r_exact)

    params_up = tsvankin_params(C_up, rho_up)
    params_low = tsvankin_params(C_low, rho_low)
    Rxz, Ryz = reflectivity(params_up, params_low, rho_up, rho_low, thetas)

    Z1 = rho_up * vp0_up
    Z2 = rho_low * np.sqrt(m_low[1] / rho_low)
    return {
        "thetas_deg": np.degrees(thetas),
        "exact": r_exact,
        "Rxz": Rxz,
        "Ryz": Ryz,
        "impedance_R0": (Z2 - Z1) / (Z2 + Z1),
        "xz_yz_diff": np.abs(Rxz - Ryz).max(),
        "max_abs_diff": np.abs(Rxz - r_exact).max(),
    }


# --------------------------------------------------------------------------
# Test cases and assertions
# --------------------------------------------------------------------------
CASES = {
    "isotropic, weak contrast": (
        dict(Vp0=3.00, Vs0=1.50, eps=0.00, delta=0.00, rho=2.30),
        dict(Vp0=3.18, Vs0=1.59, eps=0.00, delta=0.00, rho=2.38),
    ),
    "weak VTI, weak contrast": (
        dict(Vp0=3.00, Vs0=1.50, eps=0.05, delta=0.03, rho=2.30),
        dict(Vp0=3.21, Vs0=1.59, eps=0.08, delta=0.05, rho=2.38),
    ),
    "moderate VTI, moderate contrast": (
        dict(Vp0=3.00, Vs0=1.50, eps=0.12, delta=0.06, rho=2.30),
        dict(Vp0=3.60, Vs0=1.95, eps=0.20, delta=0.12, rho=2.50),
    ),
}


def main():
    results = {}
    for label, (up, low) in CASES.items():
        res = compare(up, low)
        results[label] = res
        print(f"\n=== {label} ===")
        print(f"normal incidence: exact={res['exact'][0]:+.6f}  "
              f"impedance={res['impedance_R0']:+.6f}  module={res['Rxz'][0]:+.6f}")
        print(f"max|Rxz - Ryz| (must be ~0 for VTI): {res['xz_yz_diff']:.2e}")
        print(f"{'theta(deg)':>10}{'exact':>11}{'module':>11}{'diff':>11}")
        for th, ex, mo in zip(res["thetas_deg"], res["exact"], res["Rxz"]):
            print(f"{th:10.2f}{ex:11.5f}{mo:11.5f}{mo - ex:11.5f}")
        print(f"max|module - exact| (0-35 deg): {res['max_abs_diff']:.5f}")

    print("\n--- assertions ---")
    ok = True

    def check(name, condition):
        nonlocal ok
        ok = ok and condition
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    for label, res in results.items():
        # (A) exact solver reproduces the impedance reflection at vertical
        check(f"{label}: exact R0 == impedance",
              np.isclose(res["exact"][0], res["impedance_R0"], atol=1e-9))
        # module intercept equals the impedance reflection
        check(f"{label}: module R0 == impedance",
              np.isclose(res["Rxz"][0], res["impedance_R0"], atol=1e-9))
        # VTI -> the two symmetry planes coincide
        check(f"{label}: Rxz == Ryz (VTI)", res["xz_yz_diff"] < 1e-12)

    # (B)/(convergence) tolerances scale with anisotropy & contrast strength
    check("isotropic weak: module ~ exact (<1e-3)",
          results["isotropic, weak contrast"]["max_abs_diff"] < 1e-3)
    check("weak VTI: module ~ exact (<5e-3)",
          results["weak VTI, weak contrast"]["max_abs_diff"] < 5e-3)
    # Linearisation error must grow with contrast/anisotropy, not shrink
    check("error grows with contrast/anisotropy",
          results["isotropic, weak contrast"]["max_abs_diff"]
          < results["weak VTI, weak contrast"]["max_abs_diff"]
          < results["moderate VTI, moderate contrast"]["max_abs_diff"])

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


def test_reflectivity_vs_graebner():
    """Pytest entry point: the full standalone check suite must pass."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
