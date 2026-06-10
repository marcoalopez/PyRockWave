# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: christoffel.py                                                    #
# Description: This module calculates phase and group seismic velocities      #
# in solids based on the Christoffel equation.                                #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2023-present, Marco A. Lopez-Sanchez. All rights reserved.    #
#                                                                             #
# PyRockWave is free software: you can redistribute it and/or modify          #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# PyRockWave is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with PyRockWave. If not, see <http://www.gnu.org/licenses/>.          #
#                                                                             #
# Author: Marco A. Lopez-Sanchez                                              #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
# =========================================================================== #

# Import statements
import numpy as np
import pandas as pd
from tensor_tools import _rearrange_tensor
from coordinates import sph2cart
from validation import validate_cij


# Function definitions
def phase_seismic_properties(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute phase velocities and shear-wave splitting for an array
    of propagation directions.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation, in GPa. Together
        with the density in g/cm^3 this yields velocities in km/s.

    density_gcm3 : float
        Density in g/cm^3.

    azimuths_deg : numpy.ndarray
        1D array of azimuth angles in degrees.

    polar_deg : numpy.ndarray
        1D array of polar angles in degrees, same shape as azimuths_deg.

    Returns
    -------
    tuple[pd.DataFrame, numpy.ndarray]
        df : pd.DataFrame
            One row per direction, with columns for input angles,
            wavevector components, phase velocities (Vs2, Vs1, Vp) in km/s,
            Vp/Vs ratios, shear-wave splitting percentage, and percentage
            deviations of each phase velocity from the isotropic
            Voigt-Reuss-Hill average (dVs2_perc, dVs1_perc, dVp_perc).
        eigenvectors : numpy.ndarray of shape (n, 3, 3)
            Polarization eigenvectors for each direction, indexed
            as (direction, mode, cart).
    """
    azimuths_deg, polar_deg, q, _, _, phase_vel, eigenvectors = (
        _build_christoffel_eigensystem(Cij, density_gcm3, azimuths_deg, polar_deg)
    )

    # shear wave splitting
    sws = (phase_vel[:, 1] - phase_vel[:, 0]) / ((phase_vel[:, 1] + phase_vel[:, 0]) / 2)

    # isotropic (Voigt-Reuss-Hill) reference velocities
    vp_iso, vs_iso = _isotropic_vrh_velocities(Cij, density_gcm3)

    df = pd.DataFrame(
        {
            "azimuths_deg": azimuths_deg,
            "polar_deg": polar_deg,
            "qx": q[:, 0],
            "qy": q[:, 1],
            "qz": q[:, 2],
            # phase velocities
            "Vs2_phase_kms": phase_vel[:, 0],
            "Vs1_phase_kms": phase_vel[:, 1],
            "Vp_phase_kms": phase_vel[:, 2],
            # Vp/Vs
            "VpVs1": phase_vel[:, 2] / phase_vel[:, 1],
            "VpVs2": phase_vel[:, 2] / phase_vel[:, 0],
            # polarization anisotropy (shear-wave splitting)
            "SWS_perc": 100 * sws,
            # deviation from the isotropic average
            "dVs2_perc": 100 * (phase_vel[:, 0] - vs_iso) / vs_iso,
            "dVs1_perc": 100 * (phase_vel[:, 1] - vs_iso) / vs_iso,
            "dVp_perc": 100 * (phase_vel[:, 2] - vp_iso) / vp_iso,
        }
    )

    return df, eigenvectors


def full_seismic_properties(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute phase and group velocities, enhancement factors, and
    power flow angles for an array of propagation directions.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation, in GPa. Together
        with the density in g/cm^3 this yields velocities in km/s.

    density_gcm3 : float
        Density in g/cm^3.

    azimuths_deg : numpy.ndarray
        1D array of azimuth angles in degrees.

    polar_deg : numpy.ndarray
        1D array of polar angles in degrees, same shape as azimuths_deg.

    Returns
    -------
    tuple[pd.DataFrame, numpy.ndarray]
        df : pd.DataFrame
            One row per direction, with columns for input angles,
            wavevector components, phase velocities (Vs2, Vs1, Vp) in km/s,
            group speeds and directions (per mode), group spherical angles
            (per mode), power flow angles (per mode), and enhancement
            factors (per mode).
        eigenvectors : numpy.ndarray of shape (n, 3, 3)
            Polarization eigenvectors for each direction, indexed
            as (direction, mode, cart).

    Notes
    -----
    Along and near acoustic axes (directions where the S-wave
    velocities Vs1 and Vs2 become degenerate) the eigenvalue Hessian
    is ill-defined, so the S-wave enhancement factors are unreliable
    there. See _get_hessian_eigen for details.
    """
    azimuths_deg, polar_deg, q, Cijkl_norm, eigenvalues, phase_vel, eigenvectors = (
        _build_christoffel_eigensystem(Cij, density_gcm3, azimuths_deg, polar_deg)
    )

    # COMPUTE GROUP VELOCITIES
    gradM = _christoffel_gradient_matrix(q, Cijkl_norm)
    group = calc_group_velocities(phase_vel, eigenvectors, gradM, q)

    # COMPUTE ENHANCEMENT FACTOR
    hessM = _christoffel_matrix_hessian(Cijkl_norm)
    hess_lam = _get_hessian_eigen(eigenvalues, eigenvectors, gradM, hessM)
    enh = calc_enhancement_factor(
        hess_lam, phase_vel, group["group_velocity"], group["group_speed"], q
    )

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "azimuths_deg": azimuths_deg,
            "polar_deg": polar_deg,
            "qx": q[:, 0],
            "qy": q[:, 1],
            "qz": q[:, 2],
            # phase velocities
            "Vs2_phase_kms": phase_vel[:, 0],
            "Vs1_phase_kms": phase_vel[:, 1],
            "Vp_phase_kms": phase_vel[:, 2],
            # group speeds
            "Vs2_group_kms": group["group_speed"][:, 0],
            "Vs1_group_kms": group["group_speed"][:, 1],
            "Vp_group_kms": group["group_speed"][:, 2],
            # power flow angles (already computed inside calc_group_velocities)
            "Vs2_powerflow_deg": group["powerflow_deg"][:, 0],
            "Vs1_powerflow_deg": group["powerflow_deg"][:, 1],
            "Vp_powerflow_deg": group["powerflow_deg"][:, 2],
            # enhancement factors
            "Vs2_enhancement": enh[:, 0],
            "Vs1_enhancement": enh[:, 1],
            "Vp_enhancement": enh[:, 2],
            # group spherical angles
            "Vs2_group_azimuths_deg": group["group_azimuths_deg"][:, 0],
            "Vs1_group_azimuths_deg": group["group_azimuths_deg"][:, 1],
            "Vp_group_azimuths_deg": group["group_azimuths_deg"][:, 2],
            "Vs2_group_polar_deg": group["group_polar_deg"][:, 0],
            "Vs1_group_polar_deg": group["group_polar_deg"][:, 1],
            "Vp_group_polar_deg": group["group_polar_deg"][:, 2],
        }
    )

    for mode_name, mode_idx in [("Vs2", 0), ("Vs1", 1), ("Vp", 2)]:
        df[f"{mode_name}_group_dir_x"] = group["group_dir"][:, mode_idx, 0]
        df[f"{mode_name}_group_dir_y"] = group["group_dir"][:, mode_idx, 1]
        df[f"{mode_name}_group_dir_z"] = group["group_dir"][:, mode_idx, 2]

    return df, eigenvectors


def _validate_seismic_inputs(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and coerce the inputs shared by phase_seismic_properties
    and full_seismic_properties.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    density_gcm3 : float
        Density in g/cm^3.

    azimuths_deg : numpy.ndarray
        Azimuth angles in degrees.

    polar_deg : numpy.ndarray
        Polar angles in degrees.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        azimuths_deg and polar_deg coerced to float64 arrays.

    Raises
    ------
    ValueError
        If any input fails its validation check.
    """
    validate_cij(Cij)

    azimuths_deg = np.asarray(azimuths_deg, dtype=float)
    polar_deg = np.asarray(polar_deg, dtype=float)
    if azimuths_deg.shape != polar_deg.shape:
        raise ValueError("azimuths_deg and polar_deg must have the same shape.")
    if azimuths_deg.ndim != 1:
        raise ValueError("azimuths_deg and polar_deg must be 1D arrays.")

    if not np.isfinite(density_gcm3) or density_gcm3 <= 0:
        raise ValueError("density_gcm3 must be a positive finite float.")

    return azimuths_deg, polar_deg


def _isotropic_vrh_velocities(
    Cij: np.ndarray,
    density_gcm3: float,
) -> tuple[float, float]:
    """
    Isotropic reference velocities from the Voigt-Reuss-Hill average.

    The Hill bulk (K) and shear (G) moduli are the arithmetic means
    of the Voigt and Reuss bounds, computed from the stiffness matrix
    and its inverse (compliance), respectively. The same formulas are
    used in ElasticClass.ElasticProps.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation, in GPa.

    density_gcm3 : float
        Density in g/cm^3.

    Returns
    -------
    tuple[float, float]
        (vp_iso, vs_iso), the isotropic P- and S-wave velocities
        in km/s.
    """
    Sij = np.linalg.inv(Cij)

    c11, c22, c33, c44, c55, c66 = np.diag(Cij)
    s11, s22, s33, s44, s55, s66 = np.diag(Sij)
    c12, c13, c23 = Cij[0, 1], Cij[0, 2], Cij[1, 2]
    s12, s13, s23 = Sij[0, 1], Sij[0, 2], Sij[1, 2]

    K_voigt = 1 / 9 * ((c11 + c22 + c33) + 2 * (c12 + c23 + c13))
    K_reuss = 1 / ((s11 + s22 + s33) + 2 * (s12 + s23 + s13))
    K_hill = (K_voigt + K_reuss) / 2

    G_voigt = 1 / 15 * ((c11 + c22 + c33) - (c12 + c23 + c13)
                        + 3 * (c44 + c55 + c66))
    G_reuss = 15 / (4 * (s11 + s22 + s33) - 4 * (s12 + s23 + s13)
                    + 3 * (s44 + s55 + s66))
    G_hill = (G_voigt + G_reuss) / 2

    vp_iso = np.sqrt((K_hill + 4 / 3 * G_hill) / density_gcm3)
    vs_iso = np.sqrt(G_hill / density_gcm3)

    return float(vp_iso), float(vs_iso)


def _build_christoffel_eigensystem(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate inputs, build unit wavevectors, and compute the Christoffel
    eigendecomposition and phase velocities. Shared by both public
    pipeline functions.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    density_gcm3 : float
        Density in g/cm^3.

    azimuths_deg : numpy.ndarray
        1D array of azimuth angles in degrees.

    polar_deg : numpy.ndarray
        1D array of polar angles in degrees.

    Returns
    -------
    tuple of seven numpy.ndarrays:
        azimuths_deg : (n,)   coerced float64 azimuth angles
        polar_deg    : (n,)   coerced float64 polar angles
        q            : (n, 3) unit wavevectors
        Cijkl_norm   : (3, 3, 3, 3) density-normalised elastic tensor
        eigenvalues  : (n, 3) Christoffel eigenvalues
        phase_vel    : (n, 3) phase velocities [Vs2, Vs1, Vp] in km/s
        eigenvectors : (n, 3, 3) polarization eigenvectors (mode, cart)
    """
    azimuths_deg, polar_deg = _validate_seismic_inputs(
        Cij, density_gcm3, azimuths_deg, polar_deg
    )

    azimuths_rad = np.deg2rad(azimuths_deg)
    polar_rad = np.deg2rad(polar_deg)
    x, y, z = sph2cart(azimuths_rad, polar_rad)
    q = np.column_stack((x, y, z))

    # rearrange Cij → Cijkl; normalise with density
    # Cijkl in GPa and ρ in g/cm^3 gives (km/s)^2
    Cijkl_norm = _rearrange_tensor(Cij) / density_gcm3

    M = _christoffel_matrix(q, Cijkl_norm)
    eigenvalues, eigenvectors = _calc_eigen(M)
    phase_vel = calc_phase_velocities(eigenvalues)

    return azimuths_deg, polar_deg, q, Cijkl_norm, eigenvalues, phase_vel, eigenvectors


def _christoffel_matrix(
    wavevectors: np.ndarray,
    Cijkl: np.ndarray
) -> np.ndarray:
    """
    Calculate the Christoffel matrix for a given wave vector
    and elastic tensor Cij.

    Christoffel matrix:  M_il = q_j @ C_ijkl @ q_k

    Parameters
    ----------
    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3)
        with indices (i, j, k, l).

    Returns
    -------
    numpy.ndarray
        The Christoffel matri(x)ces of shape (n, 3, 3).

    Notes
    -----
    The Christoffel matrix is calculated using the formula
    Mil = qj @ Cijkl @ qk, where Mil is the Christoffel matrix,
    q is a wave vector, and Cijkl is the elastic tensor.
    """
    if wavevectors.ndim != 2 or wavevectors.shape[1] != 3:
        raise ValueError("wavevectors must have shape (n, 3).")
    if Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl must have shape (3, 3, 3, 3).")

    return np.einsum("nj,ijkl,nk->nil", wavevectors, Cijkl, wavevectors)


def _calc_eigen(Mil: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the eigenvalues and eigenvectors of the Christoffel
    matrix sorted from low to high. The eigenvalues are related to
    primary (P) and secondary (S-fast, S-slow) wave speeds. The
    eigenvectors provide the unsigned polarization directions,
    which are orthogonal to each other. It is assumed that the
    Christoffel tensor provided is normalised to the density
    of the material.

    Parameters
    ----------
    Mil : numpy.ndarray
        The Christoffel matri(x)ces as a 3D NumPy array of
        shape (n, 3, 3).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        eigenvalues  : (n, 3) array sorted ascending per direction.
        eigenvectors : (n, 3, 3) array indexed as (direction, mode, cart).
    """
    if Mil.ndim != 3 or Mil.shape[1:] != (3, 3):
        raise ValueError("Mil must have shape (n, 3, 3).")

    eigenvalues, eigenvectors = np.linalg.eigh(Mil)  # columns are eigenvectors
    eigenvectors = np.swapaxes(eigenvectors, 1, 2)    # rows are eigenvectors (mode, cart)

    return eigenvalues, eigenvectors


def calc_phase_velocities(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute the material's sound velocities of a monochromatic
    plane wave, referred to as the phase velocity, as a function of
    crystal/aggregate orientation from the Christoffel matrix (M).
    It returns three velocities, one primary (P wave) and two
    secondary (S waves).

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized Christoffel matrix. A
        numpy array of shape (n, 3).

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Each triad contains the three wave velocities [Vs2, Vs1, Vp],
        where Vs2 < Vs1 < Vp.

    Raises
    ------
    ValueError
        If any eigenvalue is non-positive, which indicates a
        mechanically unstable elastic tensor.

    Notes
    -----
    The function estimates the phase velocities of the material's
    sound waves from the eigenvalues of the Christoffel matrix (M).
    The eigenvalues represent the squared phase velocities, and by
    taking the square root, the actual phase velocities are obtained.
    Sound waves in nature are never purely monochromatic or planar.
    See calc_group_velocities.
    """
    if eigenvalues.ndim != 2 or eigenvalues.shape[1] != 3:
        raise ValueError("eigenvalues must have shape (n, 3).")
    if np.any(eigenvalues <= 0):
        raise ValueError(
            "All Christoffel eigenvalues must be positive "
            "(mechanically stable elastic tensor)."
        )

    return np.sqrt(eigenvalues)


def _christoffel_gradient_matrix(
    wavevectors: np.ndarray,
    Cijkl: np.ndarray
) -> np.ndarray:
    """
    Gradient of Christoffel matrix with respect to
    wavevector components computed using the formula
    (e.g. Jaeken and Cottenier, 2016):

    ∂Mij / ∂q_k = ∑m (Cikmj + Cimkj) *  q_m

    Parameters
    ----------
    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray of shape (n, 3, 3, 3)
        The Christoffel gradient matrix
        gradM[n, a, i, l] = ∂M_il / ∂q_a

    Notes
    -----
    The derivative of the Christoffel matrix with respect to a wave
    vector component q_a is:
        ∂M_il/∂q_a = ∑m (C_iaml + C_imal) * q_m
    """
    if wavevectors.ndim != 2 or wavevectors.shape[1] != 3:
        raise ValueError("wavevectors must have shape (n, 3).")
    if Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl must have shape (3, 3, 3, 3).")

    q = wavevectors
    term1 = np.einsum("nm,iaml->nail", q, Cijkl)
    term2 = np.einsum("nm,imal->nail", q, Cijkl)

    return term1 + term2


def _eigenvalue_derivatives(
    eigenvectors: np.ndarray,
    gradient_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate the derivatives of eigenvalues with respect to
    the wavevector components:
        dλ_mode/dq_a = v_mode^T (∂M/∂q_a) v_mode

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape (n, 3, 3) representing three eigenvectors
        for each Christoffel matrix calculated, indexed as
        (direction, mode, cart).

    gradient_matrix : numpy.ndarray
        The derivative of the Christoffel matrix, shape (n, 3, 3, 3),
        indexed as [n, a, i, j].

    Returns
    -------
    numpy.ndarray of shape (n, 3, 3)
        dλ[n, mode, a] = derivative of eigenvalue for mode with
        respect to wavevector component a.
    """
    if eigenvectors.ndim != 3 or eigenvectors.shape[1:] != (3, 3):
        raise ValueError("eigenvectors must have shape (n, 3, 3).")
    if gradient_matrix.ndim != 4 or gradient_matrix.shape[1:] != (3, 3, 3):
        raise ValueError("gradient_matrix must have shape (n, 3, 3, 3).")

    return np.einsum("nmi,ndij,nmj->nmd", eigenvectors, gradient_matrix, eigenvectors)


def calc_spherical_angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert direction vectors to spherical angles in degrees.

    Parameters
    ----------
    vectors : numpy.ndarray of shape (..., 3)
        Direction vectors (need not be perfectly normalized). The last
        axis must contain the three Cartesian components (x, y, z).

    Returns
    -------
    polar_deg : numpy.ndarray
        Polar (colatitude) angle θ = arccos(z) in degrees, in [0°, 180°].
    azimuth_deg : numpy.ndarray
        Azimuth angle φ = arctan2(y, x) in degrees, in [0°, 360°).
    """
    v = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    x = v[..., 0]
    y = v[..., 1]
    z = np.clip(v[..., 2], -1.0, 1.0)

    theta = np.arccos(z)                  # polar
    phi = np.arctan2(y, x)                # [-pi, pi]
    phi = np.mod(phi, 2.0 * np.pi)        # [0, 2pi)

    return np.rad2deg(theta), np.rad2deg(phi)


def calc_group_velocities(
    phase_velocities: np.ndarray,
    eigenvectors: np.ndarray,
    christoffel_gradient: np.ndarray,
    wave_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate group velocity vectors, magnitudes, directions, and power flow angles.

    Matches Jaeken & Cottenier (2016) implementation logic. The power
    flow angles are computed with calc_power_flow_angles.

    Parameters
    ----------
    phase_velocities : numpy.ndarray of shape (n, 3)
        [Vs2, Vs1, Vp] phase velocities (km/s).
    eigenvectors : numpy.ndarray of shape (n, 3, 3)
        eigenvectors[n, mode, cart].
    christoffel_gradient : numpy.ndarray of shape (n, 3, 3, 3)
        gradM[n, a, i, j] = ∂M_ij/∂q_a.
    wave_vectors : numpy.ndarray of shape (n, 3)
        Unit phase directions q.

    Returns
    -------
    dict[str, numpy.ndarray] with keys:
        grad_eigenvalues  : (n, 3, 3)  dλ/dq per mode and direction
        group_velocity    : (n, 3, 3)  group velocity vectors (mode, cart)
        group_speed       : (n, 3)     magnitudes of group velocities
        group_dir         : (n, 3, 3)  unit group direction vectors
        group_polar_deg   : (n, 3)     polar angle θ of group direction
        group_azimuths_deg: (n, 3)     azimuth angle φ of group direction
        powerflow_deg     : (n, 3)     power flow angles in degrees
    """
    if phase_velocities.shape != (wave_vectors.shape[0], 3):
        raise ValueError("phase_velocities must have shape (n, 3).")
    if eigenvectors.shape != (wave_vectors.shape[0], 3, 3):
        raise ValueError("eigenvectors must have shape (n, 3, 3).")
    if christoffel_gradient.shape != (wave_vectors.shape[0], 3, 3, 3):
        raise ValueError("christoffel_gradient must have shape (n, 3, 3, 3).")
    if wave_vectors.shape[1] != 3:
        raise ValueError("wave_vectors must have shape (n, 3).")

    # dλ/dq
    grad_lam = _eigenvalue_derivatives(eigenvectors, christoffel_gradient)  # (n, mode, a)

    # v_g = (dλ/dq) / (2 v_p)
    denom = 2.0 * phase_velocities  # (n, mode)
    group_vel = grad_lam / denom[:, :, None]  # (n, mode, cart)

    group_speed = np.linalg.norm(group_vel, axis=-1)  # (n, mode)
    group_dir = group_vel / group_speed[:, :, None]

    # power flow angle between group direction and phase direction
    pf_deg = calc_power_flow_angles(group_dir, wave_vectors)

    # group direction spherical angles
    gpolar_deg, gazimuths_deg = calc_spherical_angles(group_dir)

    return {
        "grad_eigenvalues": grad_lam,
        "group_velocity": group_vel,
        "group_speed": group_speed,
        "group_dir": group_dir,
        "group_polar_deg": gpolar_deg,
        "group_azimuths_deg": gazimuths_deg,
        "powerflow_deg": pf_deg,
    }


def _christoffel_matrix_hessian(Cijkl: np.ndarray) -> np.ndarray:
    """
    Hessian of Christoffel matrix (independent of q):

        H_ab,ij = ∂²M_ij / ∂q_a ∂q_b = C_iabj + C_ibaj
        (index placement depends on chosen convention)

    This matches the Jaeken & Cottenier construction:
        hessmat[a][b][i][j] = C[i][a][b][j] + C[i][b][a][j]
    once tensor index conventions are consistent.

    Parameters
    ----------
    Cijkl : numpy.ndarray
        NumPy array of shape (3, 3, 3, 3) with indices (i, j, k, l).

    Returns
    -------
    numpy.ndarray of shape (3, 3, 3, 3)
        The Hessian of the Christoffel matrix, hessM[a, b, i, j].
    """
    if Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl must have shape (3, 3, 3, 3).")

    return np.transpose(Cijkl, (1, 2, 0, 3)) + np.transpose(Cijkl, (2, 1, 0, 3))


def _get_hessian_eigen(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    gradient_matrix: np.ndarray,
    hess_matrix: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Hessian of eigenvalues (per direction and mode):

        Hess(λ_mode) = v^T (Hess(M)) v  +  2 * (dM v) * pinv * (dM v)^T

    This follows Jaeken & Cottenier (2016) implementation.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized Christoffel matrices,
        shape (n, 3).

    eigenvectors : numpy.ndarray
        The eigenvectors of the normalized Christoffel matrices,
        shape (n, 3, 3), indexed as (direction, mode, cart).

    gradient_matrix : numpy.ndarray
        The derivatives of the Christoffel matrices, shape (n, 3, 3, 3),
        indexed as [n, a, i, j].

    hess_matrix : numpy.ndarray
        The Hessian of the Christoffel matrix, shape (3, 3, 3, 3),
        indexed as [a, b, i, j].

    tol : float, optional
        Degeneracy tolerance for eigenvalue differences, in the same
        (absolute) units as the eigenvalues, i.e. (km/s)^2. Default
        is 1e-10.

    Returns
    -------
    numpy.ndarray of shape (n, 3, 3, 3)
        Hlam[n, mode, a, b] = ∂²λ_mode / ∂q_a ∂q_b

    Notes
    -----
    Second-order perturbation terms are dropped whenever the
    eigenvalue gap |λ_m - λ_i| is below `tol`. Eigenvalues are not
    twice differentiable at degeneracies, so along and near acoustic
    axes (where Vs1 = Vs2) the returned S-mode Hessians, and any
    quantity derived from them such as the enhancement factor, are
    unreliable. This matches the behaviour of the Jaeken & Cottenier
    (2016) reference implementation.
    """
    n = eigenvalues.shape[0]

    # Term 1: v^T Hess(M) v  -> (n, mode, a, b)
    term1 = np.einsum("abij,nmi,nmj->nmab", hess_matrix, eigenvectors, eigenvectors)

    # Pseudoinverse term: pinv = V^T diag(1/(λ_m-λ_i)) V, with diag_mm = 0
    denom = eigenvalues[:, :, None] - eigenvalues[:, None, :]  # (n, mode_target, mode_i)
    w = np.zeros_like(denom)
    mask = np.abs(denom) >= tol
    w[mask] = 1.0 / denom[mask]

    diagmat = np.zeros((n, 3, 3, 3), dtype=eigenvalues.dtype)  # (n, mode_target, 3, 3)
    idx = np.arange(3)
    diagmat[:, :, idx, idx] = w

    V = eigenvectors                       # (n, mode, cart)
    VT = np.transpose(V, (0, 2, 1))        # (n, cart, mode)
    tmp = VT[:, None, :, :] @ diagmat      # (n, mode_target, cart, mode)
    pinv = tmp @ V[:, None, :, :]          # (n, mode_target, cart, cart)

    # deriv_vec = gradM @ v  -> (n, mode, a, cart)
    deriv = np.einsum("naij,nmj->nmai", gradient_matrix, eigenvectors)

    # Term 2: 2 * deriv * pinv * deriv^T  -> (n, mode, a, b)
    term2 = 2.0 * np.einsum("nmai,nmij,nmbj->nmab", deriv, pinv, deriv)

    return term1 + term2


def _validate_calc_enhancement_factor(
    hessian_eigenvalues: np.ndarray,
    phase_velocities: np.ndarray,
    group_velocity: np.ndarray,
    group_speed: np.ndarray,
    wave_vectors: np.ndarray,
) -> None:
    """
    Validate the array shapes passed to calc_enhancement_factor.

    Parameters
    ----------
    See calc_enhancement_factor.

    Raises
    ------
    ValueError
        If any array does not have the expected shape.
    """
    if wave_vectors.ndim != 2 or wave_vectors.shape[1] != 3:
        raise ValueError("wave_vectors must have shape (n, 3).")
    n = wave_vectors.shape[0]
    if hessian_eigenvalues.shape != (n, 3, 3, 3):
        raise ValueError("hessian_eigenvalues must have shape (n, 3, 3, 3).")
    if phase_velocities.shape != (n, 3):
        raise ValueError("phase_velocities must have shape (n, 3).")
    if group_velocity.shape != (n, 3, 3):
        raise ValueError("group_velocity must have shape (n, 3, 3).")
    if group_speed.shape != (n, 3):
        raise ValueError("group_speed must have shape (n, 3).")


def calc_enhancement_factor(
    hessian_eigenvalues: np.ndarray,
    phase_velocities: np.ndarray,
    group_velocity: np.ndarray,
    group_speed: np.ndarray,
    wave_vectors: np.ndarray,
) -> np.ndarray:
    """
    Enhancement factor (exact, analytical) as in Jaeken & Cottenier (2016).

    The enhancement factor measures how energy is focused or defocused
    relative to an isotropic medium; values > 1 indicate focusing.

    Parameters
    ----------
    hessian_eigenvalues : numpy.ndarray of shape (n, 3, 3, 3)
        Hlam[n, mode, a, b] = ∂²λ_mode / ∂q_a ∂q_b.
    phase_velocities : numpy.ndarray of shape (n, 3)
        Phase velocities [Vs2, Vs1, Vp] in km/s.
    group_velocity : numpy.ndarray of shape (n, 3, 3)
        Group velocity vectors, vg[n, mode, cart].
    group_speed : numpy.ndarray of shape (n, 3)
        Magnitudes of group velocities |vg|.
    wave_vectors : numpy.ndarray of shape (n, 3)
        Unit phase directions q.

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Enhancement factor per direction and wave mode.

    Notes
    -----
    The S-mode enhancement factors are unreliable along and near
    acoustic axes (degenerate Vs1 = Vs2 directions); see
    _get_hessian_eigen. Directions at an exact caustic return inf.
    """
    _validate_calc_enhancement_factor(
        hessian_eigenvalues, phase_velocities, group_velocity,
        group_speed, wave_vectors,
    )

    H = hessian_eigenvalues
    vp = phase_velocities
    vg = group_velocity
    vg_abs = group_speed
    q = wave_vectors

    # grad_group = d(vg)/dq from paper's algebra (implemented in reference code)
    Hv = np.einsum("nmab,nmb->nma", H, vg)              # (n, mode, a)
    outer = vg[..., :, None] * Hv[..., None, :]         # (n, mode, a, b)

    grad_group = H / vg_abs[..., None, None] - outer / (vg_abs[..., None, None] ** 3)
    grad_group = grad_group / (2.0 * vp[..., None, None])

    cof = _cofactor_3x3(grad_group)                     # (n, mode, 3, 3)
    vec = np.einsum("nmij,nj->nmi", cof, q)             # (n, mode, 3)

    denom = np.linalg.norm(vec, axis=-1)
    return np.divide(
        1.0, denom, out=np.full_like(denom, np.inf), where=denom > 0.0
    )


def calc_power_flow_angles(
    group_dir: np.ndarray,
    wave_vectors: np.ndarray,
) -> np.ndarray:
    """
    Power flow angle (degrees) between group direction
    and phase direction.

    Parameters
    ----------
    group_dir : numpy.ndarray of shape (n, 3, 3)
        Unit group directions, group_dir[n, mode, cart].
    wave_vectors : numpy.ndarray of shape (n, 3)
        Unit phase directions.

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Power flow angles in degrees per direction and wave mode.
    """
    if group_dir.ndim != 3 or group_dir.shape[1:] != (3, 3):
        raise ValueError("group_dir must have shape (n, 3, 3).")
    if wave_vectors.ndim != 2 or wave_vectors.shape[1] != 3:
        raise ValueError("wave_vectors must have shape (n, 3).")

    cosang = np.einsum("nmi,ni->nm", group_dir, wave_vectors)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.rad2deg(np.arccos(cosang))


# =================================================================
# Private helpers for internal use only

def _cofactor_3x3(A: np.ndarray) -> np.ndarray:
    """
    Vectorized cofactor matrix for arrays of 3x3 matrices.

    Parameters
    ----------
    A : numpy.ndarray of shape (..., 3, 3)
        Input array; the last two axes must be 3x3.

    Returns
    -------
    numpy.ndarray of shape (..., 3, 3)
        Cofactor matrices.
    """
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]

    cof = np.empty_like(A)
    cof[..., 0, 0] = e * i - f * h
    cof[..., 0, 1] = f * g - d * i
    cof[..., 0, 2] = d * h - e * g
    cof[..., 1, 0] = c * h - b * i
    cof[..., 1, 1] = a * i - c * g
    cof[..., 1, 2] = b * g - a * h
    cof[..., 2, 0] = b * f - c * e
    cof[..., 2, 1] = c * d - a * f
    cof[..., 2, 2] = a * e - b * d

    return cof

# End of file
