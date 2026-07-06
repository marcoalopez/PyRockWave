"""Independent validation of layered_media.zoeppritz_reflectivity().

The exact anisotropic reflection/transmission solver is checked four ways:

  (1) Exactness (VTI): against the independent Graebner (1992) exact VTI
      P-SV solver implemented in test_reflectivity_vs_graebner.py. Both are
      exact, so they must agree to numerical precision (isotropic media are
      covered as the eps = delta = 0 special case, i.e. classical Zoeppritz).
  (2) Physics anchors: at normal incidence Rpp must equal the acoustic
      impedance reflection (Z2 - Z1) / (Z2 + Z1), and for VTI media the
      result must be independent of azimuth.
  (3) Convergence (orthorhombic): for weak contrast and weak anisotropy it
      must converge to the Ruger (1998) linearisation implemented in
      calc_reflectivity(), in both symmetry planes (azimuth 0 and 90).
  (4) Energy conservation: the vertical energy flux of the incident wave
      must equal the summed fluxes of all propagating scattered waves, both
      below and beyond the critical angle(s), including a strongly
      anisotropic (olivine) lower medium. This exercises the complex
      post-critical branch, which the Graebner comparison cannot reach.

Run standalone (no pytest dependency):
    python tests/test_zoeppritz_reflectivity.py

References
----------
Graebner, M., 1992, Plane-wave reflection and transmission coefficients for a
    transversely isotropic solid: Geophysics, 57, 1512-1519.
Schoenberg, M., and Protazio, J., 1992, 'Zoeppritz' rationalized and
    generalized to anisotropy: Journal of Seismic Exploration, 1, 125-144.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pyrockwave.layered_media import (  # noqa: E402
    calc_reflectivity,
    zoeppritz_reflectivity,
    _select_scattered_modes,
    _upgoing_mask,
    _vertical_slowness_modes,
)
from pyrockwave.utils.coordinates import equispaced_S2_grid  # noqa: E402
from pyrockwave.utils.tensor_tools import _rearrange_tensor  # noqa: E402
from pyrockwave.christoffel import _calc_eigen, _christoffel_matrix  # noqa: E402
from test_reflectivity_vs_graebner import vti_exact_rpp, vti_tensor  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def complex_coeff(df, name):
    """Rebuild the complex coefficient from the magnitude/phase columns."""
    mag = df[f"{name}_mag"].to_numpy()
    phase = np.deg2rad(df[f"{name}_phase_deg"].to_numpy())
    return mag * np.exp(1j * phase)


def vti_qp_phase_velocity(c11, c33, c44, c13, rho, theta_rad):
    """Exact VTI qP phase velocity at phase angle theta (closed form)."""
    s2, c2 = np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2
    m = (c11 - c44) * s2 - (c33 - c44) * c2
    root = np.sqrt(m**2 + 4 * (c13 + c44) ** 2 * s2 * c2)
    return np.sqrt(((c11 + c44) * s2 + (c33 + c44) * c2 + root) / (2 * rho))


def ortho_tensor(Vp0, Vs0, eps1, d1, eps2, d2, g1, g2, d3, rho):
    """Orthorhombic stiffness (GPa) from Tsvankin parameters, inverting the
    definitions used by anisotropic_models.tsvankin_params()."""
    c33 = rho * Vp0**2
    c55 = rho * Vs0**2
    c11 = c33 * (1 + 2 * eps2)
    c22 = c33 * (1 + 2 * eps1)
    c66 = c55 * (1 + 2 * g1)
    c44 = c66 / (1 + 2 * g2)
    c13 = -c55 + np.sqrt(2 * c33 * (c33 - c55) * d2 + (c33 - c55) ** 2)
    c23 = -c44 + np.sqrt(2 * c33 * (c33 - c44) * d1 + (c33 - c44) ** 2)
    c12 = -c66 + np.sqrt(2 * c11 * (c11 - c66) * d3 + (c11 - c66) ** 2)
    return np.array([
        [c11, c12, c13,   0,   0,   0],
        [c12, c22, c23,   0,   0,   0],
        [c13, c23, c33,   0,   0,   0],
        [  0,   0,   0, c44,   0,   0],
        [  0,   0,   0,   0, c55,   0],
        [  0,   0,   0,   0,   0, c66],
    ])


def isotropic_tensor(Vp, Vs, rho):
    """Isotropic stiffness (GPa) from velocities (km/s) and density."""
    c33 = rho * Vp**2
    c44 = rho * Vs**2
    c12 = c33 - 2 * c44
    C = np.diag([c33, c33, c33, c44, c44, c44]).astype(float)
    C[:3, :3] += c12 * (1 - np.eye(3))
    return C


# --------------------------------------------------------------------------
# (1) + (2) Exact VTI cross-check, impedance anchor, azimuthal invariance
# --------------------------------------------------------------------------
def check_vti_exact(upper, lower, theta_max_deg, azimuth_deg=0.0, n=25):
    """Max |Rpp difference| between the module and the independent Graebner
    solver (pre-critical angles only), plus the normal-incidence anchor."""
    C_up, m_up = vti_tensor(**upper)
    C_low, m_low = vti_tensor(**lower)
    rho_up, rho_low = upper["rho"], lower["rho"]

    thetas_deg = np.linspace(0.0, theta_max_deg, n)
    df = zoeppritz_reflectivity(
        C_up, C_low, rho_up, rho_low,
        np.full_like(thetas_deg, azimuth_deg), thetas_deg,
    )
    rpp_module = complex_coeff(df, "Rpp")

    rpp_exact = []
    for theta in np.deg2rad(thetas_deg):
        vp = vti_qp_phase_velocity(*m_up, theta)
        rpp, _ = vti_exact_rpp(m_up, m_low, np.sin(theta) / vp)
        rpp_exact.append(rpp)
    rpp_exact = np.array(rpp_exact)

    Z1 = rho_up * np.sqrt(m_up[1] / rho_up)
    Z2 = rho_low * np.sqrt(m_low[1] / rho_low)
    return {
        "max_diff": np.abs(rpp_module - rpp_exact).max(),
        "R0_error": abs(rpp_module[0].real - (Z2 - Z1) / (Z2 + Z1)),
        "max_imag_precritical": np.abs(rpp_module.imag).max(),
    }


def check_azimuthal_invariance(upper, lower, theta_deg=25.0):
    """For VTI media the response must not depend on azimuth."""
    C_up, _ = vti_tensor(**upper)
    C_low, _ = vti_tensor(**lower)
    azimuths = np.array([0.0, 37.0, 90.0, 210.0])
    df = zoeppritz_reflectivity(
        C_up, C_low, upper["rho"], lower["rho"],
        azimuths, np.full_like(azimuths, theta_deg),
    )
    rpp = complex_coeff(df, "Rpp")
    return np.abs(rpp - rpp[0]).max()


# --------------------------------------------------------------------------
# (3) Convergence to the Ruger linearisation (orthorhombic, weak)
# --------------------------------------------------------------------------
def check_ruger_convergence(theta_max_deg=30.0, n=13):
    upper = dict(Vp0=3.00, Vs0=1.50, eps1=0.03, d1=0.02, eps2=0.05, d2=0.03,
                 g1=0.02, g2=0.04, d3=0.02, rho=2.30)
    lower = dict(Vp0=3.15, Vs0=1.57, eps1=0.05, d1=0.03, eps2=0.07, d2=0.04,
                 g1=0.03, g2=0.05, d3=0.03, rho=2.38)
    C_up = ortho_tensor(**upper)
    C_low = ortho_tensor(**lower)
    rho_up, rho_low = upper["rho"], lower["rho"]

    thetas_deg = np.linspace(0.0, theta_max_deg, n)
    Rxz_ruger, Ryz_ruger = calc_reflectivity(
        C_up, C_low, rho_up, rho_low, thetas_deg
    )

    df_xz = zoeppritz_reflectivity(
        C_up, C_low, rho_up, rho_low, np.zeros_like(thetas_deg), thetas_deg
    )
    df_yz = zoeppritz_reflectivity(
        C_up, C_low, rho_up, rho_low, np.full_like(thetas_deg, 90.0), thetas_deg
    )
    return {
        "xz": np.abs(complex_coeff(df_xz, "Rpp").real - Rxz_ruger).max(),
        "yz": np.abs(complex_coeff(df_yz, "Rpp").real - Ryz_ruger).max(),
    }


# --------------------------------------------------------------------------
# (4) Energy conservation (incl. post-critical, strong anisotropy)
# --------------------------------------------------------------------------
def check_energy_conservation(C_up, C_low, rho_up, rho_low,
                              azimuth_deg, thetas_deg):
    """Max relative violation of |F_inc| = sum of scattered vertical energy
    fluxes. Rebuilds the scattered wavefields with the module's private
    helpers and weights each flux by |coefficient|^2; evanescent waves carry
    no net flux and are excluded (their |Im q| makes the flux vanish)."""
    df = zoeppritz_reflectivity(
        C_up, C_low, rho_up, rho_low,
        np.full_like(thetas_deg, azimuth_deg), thetas_deg,
    )
    coeffs = np.column_stack(
        [complex_coeff(df, name)
         for name in ("Rpp", "Rps1", "Rps2", "Tpp", "Tps1", "Tps2")]
    )

    # Rebuild the incident and scattered plane waves (same path as module)
    az, po = np.deg2rad(azimuth_deg), np.deg2rad(thetas_deg)
    wavevectors = np.column_stack(
        (np.sin(po) * np.cos(az), np.sin(po) * np.sin(az), -np.cos(po))
    )
    chat_up = _rearrange_tensor(C_up) / rho_up
    chat_low = _rearrange_tensor(C_low) / rho_low

    eigenvalues, eigenvectors = _calc_eigen(
        _christoffel_matrix(wavevectors, chat_up)
    )
    pol_inc = eigenvectors[:, 2, :]
    slow_inc = wavevectors / np.sqrt(eigenvalues[:, 2])[:, np.newaxis]
    slow_h = slow_inc.copy()
    slow_h[:, 2] = 0.0

    def flux_z(chat, rho, pol, slow):
        """rho * Re(chat_3jkl a*_j a_k s_l), per mode (n, m)."""
        return rho * np.real(
            np.einsum("jkl,nmj,nmk,nml->nm", chat[2], np.conj(pol), pol, slow)
        )

    q1, p1, s1 = _vertical_slowness_modes(chat_up, slow_h)
    q2, p2, s2 = _vertical_slowness_modes(chat_low, slow_h)
    pol_r, slow_r = _select_scattered_modes(
        chat_up, p1, s1, _upgoing_mask(chat_up, q1, p1, s1), "reflected"
    )
    pol_t, slow_t = _select_scattered_modes(
        chat_low, p2, s2, ~_upgoing_mask(chat_low, q2, p2, s2), "transmitted"
    )

    f_inc = flux_z(chat_up, rho_up, pol_inc[:, None, :], slow_inc[:, None, :])[:, 0]
    f_scat = np.concatenate(
        (flux_z(chat_up, rho_up, pol_r, slow_r),
         flux_z(chat_low, rho_low, pol_t, slow_t)), axis=1
    )
    scattered = np.sum(np.abs(coeffs) ** 2 * np.abs(f_scat), axis=1)
    return np.abs(scattered - np.abs(f_inc)).max() / np.abs(f_inc).max()


# --------------------------------------------------------------------------
# Test cases and assertions
# --------------------------------------------------------------------------
VTI_CASES = {
    "isotropic, strong contrast": (
        dict(Vp0=3.00, Vs0=1.50, eps=0.00, delta=0.00, rho=2.30),
        dict(Vp0=4.00, Vs0=2.20, eps=0.00, delta=0.00, rho=2.60),
        40.0,  # P critical angle ~48.6 deg
    ),
    "moderate VTI, moderate contrast": (
        dict(Vp0=3.00, Vs0=1.50, eps=0.12, delta=0.06, rho=2.30),
        dict(Vp0=3.60, Vs0=1.95, eps=0.20, delta=0.12, rho=2.50),
        40.0,
    ),
    "strong VTI, strong contrast": (
        dict(Vp0=3.00, Vs0=1.60, eps=0.25, delta=0.10, rho=2.30),
        dict(Vp0=3.90, Vs0=2.10, eps=0.15, delta=0.08, rho=2.65),
        35.0,
    ),
}


def main():
    ok = True

    def check(name, condition):
        nonlocal ok
        ok = ok and condition
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    print("=== (1)/(2) exact VTI cross-check (Graebner 1992) ===")
    for label, (up, low, theta_max) in VTI_CASES.items():
        res = check_vti_exact(up, low, theta_max)
        print(f"{label}: max|Rpp_module - Rpp_exact| = {res['max_diff']:.2e}, "
              f"R0 error = {res['R0_error']:.2e}")
        check(f"{label}: matches exact solver (<1e-8)", res["max_diff"] < 1e-8)
        check(f"{label}: R0 == impedance (<1e-9)", res["R0_error"] < 1e-9)
        check(f"{label}: real pre-critical (<1e-9)",
              res["max_imag_precritical"] < 1e-9)

    print("\n=== (2) azimuthal invariance (VTI) ===")
    up, low, _ = VTI_CASES["moderate VTI, moderate contrast"]
    az_dev = check_azimuthal_invariance(up, low)
    print(f"max deviation across azimuths: {az_dev:.2e}")
    check("VTI: Rpp independent of azimuth (<1e-9)", az_dev < 1e-9)

    print("\n=== (3) convergence to Ruger (orthorhombic, weak) ===")
    res = check_ruger_convergence()
    print(f"max|exact - Ruger|: xz = {res['xz']:.2e}, yz = {res['yz']:.2e}")
    check("xz plane: exact ~ Ruger (<5e-3)", res["xz"] < 5e-3)
    check("yz plane: exact ~ Ruger (<5e-3)", res["yz"] < 5e-3)

    print("\n=== (4) energy conservation ===")
    # isotropic strong contrast, sweep through both critical angles
    C_up = isotropic_tensor(2.0, 1.0, 2.2)
    C_low = isotropic_tensor(4.0, 2.5, 2.7)  # P crit 30 deg, S crit ~53 deg
    thetas = np.linspace(0.0, 85.0, 86)
    thetas = thetas[np.abs(thetas - 30.0) > 0.5]  # avoid exact criticality
    err_iso = check_energy_conservation(C_up, C_low, 2.2, 2.7, 20.0, thetas)
    print(f"isotropic incl. post-critical: max rel. error = {err_iso:.2e}")
    check("energy conserved, isotropic (<1e-8)", err_iso < 1e-8)

    # isotropic over olivine (strong triclinic-handled orthorhombic case)
    C_olivine = np.array([
        [320.5,  68.1,  71.6,   0.0,   0.0,   0.0],
        [ 68.1, 196.5,  76.8,   0.0,   0.0,   0.0],
        [ 71.6,  76.8, 233.5,   0.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,  64.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,  77.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,   0.0,  78.7],
    ])
    C_up = isotropic_tensor(7.0, 3.8, 2.9)
    err_oli = check_energy_conservation(
        C_up, C_olivine, 2.9, 3.355, 30.0, np.linspace(0.0, 85.0, 86)
    )
    print(f"isotropic/olivine incl. post-critical: max rel. error = {err_oli:.2e}")
    check("energy conserved, olivine (<1e-8)", err_oli < 1e-8)

    print("\n=== (5) hemispheric orientation grid (smoke test) ===")
    azimuths, polar = equispaced_S2_grid(
        num_points=500, degrees=True, hemisphere="upper"
    )
    subhorizontal = polar < 90.0  # grazing incidence is not defined
    df = zoeppritz_reflectivity(
        C_up, C_olivine, 2.9, 3.355,
        azimuths[subhorizontal], polar[subhorizontal],
    )
    print(f"grid points evaluated: {len(df)}")
    check("grid output has one row per direction",
          len(df) == subhorizontal.sum())
    check("grid output is finite",
          bool(np.isfinite(df.to_numpy()).all()))

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
