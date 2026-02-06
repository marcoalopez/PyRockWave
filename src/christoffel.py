###############################################################################
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: christoffel.py                                                    #
# Description: This module calculates phase and group seismic velocities      #
# in solids based on the Christoffel equation.                                #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2023-2025, Marco A. Lopez-Sanchez. All rights reserved.       #
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
###############################################################################

# Import statements
import numpy as np
import pandas as pd
from tensor_tools import _rearrange_tensor
from coordinates import sph2cart


# Function definitions
def phase_seismic_properties(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> pd.DataFrame:
    """
    Compute phase/group velocities, enhancement factors,
    and power flow angles.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.
    
    density : float
        density in g/cm^3

    azimuths_deg, phi_deg : numpy.ndarray
        1D arrays describing wavevector directions

    Returns
    -------
    pd.DataFrame
        One row per direction, with columns for:
        - input angles, wavevector components
        - phase velocities (Vs2, Vs1, Vp)
        - group speed and direction (per mode)
        - group spherical angles (per mode)
        - power flow angles (per mode)
        - enhancement factors (per mode)
    """
    # Sanity checks on inputs
    validate_cijs(Cij)

    azimuths_deg = np.asarray(azimuths_deg, dtype=float)
    polar_deg = np.asarray(polar_deg, dtype=float)
    if azimuths_deg.shape != polar_deg.shape:
        raise ValueError("azimuths_deg and polar_deg must have the same shape.")
    if azimuths_deg.ndim != 1:
        raise ValueError("azimuths_deg and polar_deg must be 1D arrays.")

    if not np.isfinite(density_gcm3) or density_gcm3 <= 0:
        raise ValueError("density_gcm3 must be a positive finite float.")

    # Build unit wavevectors q (n, 3)
    azimuths_rad = np.deg2rad(azimuths_deg)
    polar_rad = np.deg2rad(polar_deg)
    x, y, z = sph2cart(azimuths_rad, polar_rad)
    q = np.column_stack((x, y, z))

    # rearrange tensor Cij → Cijkl
    Cijkl = _rearrange_tensor(Cij)

    # normalize Cijkl with density, with Cijkl in GPa and ρ in g/cm^3 gives (km/s)^2
    Cijkl_norm = Cijkl / density_gcm3

    # estimate the Christoffel matrix (M) for every wavevector (n,3,3)
    M = _christoffel_matrix(q, Cijkl_norm)

    # estimate the eigenvalues and eigenvectors, (n,3), (n,3,3)
    eigenvalues, eigenvectors = _calc_eigen(M)

    # COMPUTE PHASE VELOCITIES   
    phase_vel = calc_phase_velocities(eigenvalues)  # (n,3)

    # compute shear wave splitting
    sws = (phase_vel[:, 1] - phase_vel[:, 0]) / ((phase_vel[:, 1] + phase_vel[:, 0]) / 2)

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
            # Vp/Vs
            "VpVs1": phase_vel[:, 2] / phase_vel[:, 1],
            "VpVs2": phase_vel[:, 2] / phase_vel[:, 0],
            # polarization anisotropy (shear-wave splitting)
            "SWS_perc": 100 * sws,
            # deviation from the isotropic average
            # TODO
        }
    )

    return df, eigenvectors

def full_seismic_properties(
    Cij: np.ndarray,
    density_gcm3: float,
    azimuths_deg: np.ndarray,
    polar_deg: np.ndarray,
) -> pd.DataFrame:
    """
    Compute phase/group velocities, enhancement factors,
    and power flow angles.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.
    
    density : float
        density in g/cm^3

    azimuths_deg, phi_deg : numpy.ndarray
        1D arrays describing wavevector directions

    Returns
    -------
    pd.DataFrame
        One row per direction, with columns for:
        - input angles, wavevector components
        - phase velocities (Vs2, Vs1, Vp)
        - group speed and direction (per mode)
        - group spherical angles (per mode)
        - power flow angles (per mode)
        - enhancement factors (per mode)
    """
    # Sanity checks on inputs
    validate_cijs(Cij)

    azimuths_deg = np.asarray(azimuths_deg, dtype=float)
    polar_deg = np.asarray(polar_deg, dtype=float)
    if azimuths_deg.shape != polar_deg.shape:
        raise ValueError("azimuths_deg and polar_deg must have the same shape.")
    if azimuths_deg.ndim != 1:
        raise ValueError("azimuths_deg and polar_deg must be 1D arrays.")

    if not np.isfinite(density_gcm3) or density_gcm3 <= 0:
        raise ValueError("density_gcm3 must be a positive finite float.")

    # Build unit wavevectors q (n, 3)
    azimuths_rad = np.deg2rad(azimuths_deg)
    polar_rad = np.deg2rad(polar_deg)
    x, y, z = sph2cart(azimuths_rad, polar_rad)
    q = np.column_stack((x, y, z))

    # rearrange tensor Cij → Cijkl
    Cijkl = _rearrange_tensor(Cij)

    # normalize Cijkl with density
    # with Cijkl in GPa and density in g/cm^3 gives (km/s)^2
    Cijkl_norm = Cijkl / density_gcm3

    # estimate the Christoffel matrix (M) for every wavevector (n,3,3)
    M = _christoffel_matrix(q, Cijkl_norm)

    # estimate the eigenvalues and eigenvectors, (n,3), (n,3,3)
    eigenvalues, eigenvectors = _calc_eigen(M)

    # COMPUTE PHASE VELOCITIES   
    phase_vel = calc_phase_velocities(eigenvalues)  # (n,3)

    # COMPUTE GROUP VELOCITIES
    gradM = _christoffel_gradient_matrix(q, Cijkl_norm)
    group = calc_group_velocities(phase_vel, eigenvectors, gradM, q)

    # COMPUTE ENHANCEMENT FACTOR
    hessM = _christoffel_matrix_hessian(Cijkl_norm)
    Hλ = _get_hessian_eigen(eigenvalues, eigenvectors, gradM, hessM)
    enh = calc_enhancement_factor(
        Hλ, phase_vel, group["group_velocity"], group["group_speed"], q
    )

    # Power flow (already computed in group dict, but keep a dedicated function too)
    pf_deg = calc_power_flow_angles(group["group_dir"], q)

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
            # power flow angles
            "Vs2_powerflow_deg": pf_deg[:, 0],
            "Vs1_powerflow_deg": pf_deg[:, 1],
            "Vp_powerflow_deg": pf_deg[:, 2],
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

    # Optional: add group direction components (often useful)
    for mode_name, mode_idx in [("Vs2", 0), ("Vs1", 1), ("Vp", 2)]:
        df[f"{mode_name}_group_dir_x"] = group["group_dir"][:, mode_idx, 0]
        df[f"{mode_name}_group_dir_y"] = group["group_dir"][:, mode_idx, 1]
        df[f"{mode_name}_group_dir_z"] = group["group_dir"][:, mode_idx, 2]

    return df, eigenvectors


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

    return np.einsum("nj,ijkl,nk->nil", wavevectors, Cijkl, wavevectors)


def _calc_eigen(Mil: np.ndarray):
    """Return the eigenvalues and eigenvectors of the Christoffel
    matrix sorted from low to high. The eigenvalues are related to
    primary (P) and secondary (S-fast, S-slow) wave speeds. The
    eigenvectors provide the unsigned polarization directions,
    which are orthogonal to each other. It is assumed that the
    Christoffel tensor provided is normalised to the density
    of the material.

    Parameters
    ----------
    Mij : numpy.ndarray
        the Christoffel matri(x)ces as a 2D NumPy array of
        shape (n, 3, 3).

    Returns
    -------
    two numpy.ndarrays
        a (n, 3) array with the eigenvalues of each matrix
        a (n, 3, 3) array with the eigenvectors of each matrix
    """

    eigen_values, eigen_vectors = np.linalg.eigh(Mil)  # v columns are eigenvectors
    # eigen_vectors = np.swapaxes(eigen_vectors, 1, 2)   # rows are eigenvectors (mode, cart)

    return eigen_values, eigen_vectors


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
        numpy array of shape (n, 3)

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Each triad contains the three wave velocities [Vs2, Vs1, Vp],
        where Vs2 < Vs1 < Vp.

    Notes
    -----
    The function estimates the phase velocities of the material's
    sound waves from the eigenvalues of the Christoffel matrix (M).
    The eigenvalues represent the squared phase velocities, and by
    taking the square root, the actual phase velocities are obtained.
    The output is a 1D NumPy array containing the three velocities,
    Vs2, Vs1, and Vp, sorted in ascending order (Vs2 < Vs1 < Vp).
    Sound waves in nature are never purely monochromatic or planar.
    See calc_group_velocities.
    """

    # get the number of eigen values to proccess
    n = eigenvalues.shape[0]

    # preallocate array
    phase_speeds = np.zeros((n, 3))

    # TO REIMPLEMENT ASSUMING A SHAPE (n, 3)
    for i in range(n):
        phase_speeds[i, :] = np.sign(eigenvalues[i, :]) * np.sqrt(
            np.absolute(eigenvalues[i, :])
        )

    return phase_speeds


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
    numpy.ndarray (n, 3, 3, 3)
        The Christoffel gradient matrix
        gradM[n, a, i, l] = ∂M_il / ∂q_a

    Notes
    -----
    The gradient matrix gradmat[n, i, j] is computed using the wave
    vector and the elastic tensor Cijkl. The derivative of the
    Christoffel matrix with respect a wave vector is given by the
    summation of the products of q_k (wave vector component)
    and the terms (Cikmj + Cimkj). The final result is a
    3D matrix with shape (3, 3, 3).
    """

    q = wavevectors
    term1 = np.einsum("nm,iaml->nail", q, Cijkl)
    term2 = np.einsum("nm,imal->nail", q, Cijkl)

    return term1 + term2


def _eigenvalue_derivatives(
    eigenvectors: np.ndarray,
    gradient_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate the derivatives of eigenvectors with respect to
    the gradient matrix:
        dλ_mode/dq_a = v_mode^T (∂M/∂q_a) v_mode

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape (n, 3, 3) representing three eigenvectors
        for each Christoffel matrix calculated, rows=modes.

    gradient_matrix : numpy.ndarray
        The derivative of the Christoffel matrix, which has a
        shape of (n, 3, 3, 3) [a, i, j]

    Returns
    -------
    numpy.ndarray:
        Array of shape (n, 3, 3), where the i-th index
        corresponds to the i-th eigenvector, and each element
        (a, i, j) represents the derivative of the a-th
        eigenvector with respect to the i-th component of
        the j-th eigenvector. [n, mode, a]
    """

    if eigenvectors.ndim != 3 or eigenvectors.shape[1:] != (3, 3):
        raise ValueError("eigenvectors must have shape (n, 3, 3).")
    if gradient_matrix.ndim != 4 or gradient_matrix.shape[1:] != (3, 3, 3):
        raise ValueError("gradient_matrix must have shape (n, 3, 3, 3).")

    return np.einsum("nmi,ndij,nmj->nmd", eigenvectors, gradient_matrix, eigenvectors)


def group_velocities(
    phase_velocities: np.ndarray,
    eigenvalue_gradients: np.ndarray,
    wave_vectors: np.ndarray,
):
    """
    Calculate the group velocities for seismic waves in given
    directions, and their magnitudes and directions.

    Parameters
    ----------
    phase_velocities : np.ndarray of shape (n, 3)
        Phase velocities for n directions and 3 wave
        polarizations (P, S1, S2).

    eigenvalue_gradients : np.ndarray of shape (n, 3, 3)
        Gradients of phase velocities with respect to wavenumber
        for n directions, 3 polarizations, and 3 spatial dimensions.

    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Returns
    -------
    tuple containing:
        group_velocities : np.ndarray of shape (n, 3, 3)
            Group velocities for n directions, 3 wave polarizations,
            and 3 spatial dimensions.
        group_velocity_magnitudes : np.ndarray of shape (n, 3)
            Magnitudes of group velocities for n directions and
            3 wave polarizations.
        group_velocity_directions : np.ndarray of shape (n, 3, 3)
            Unit vectors representing directions of group velocities
            for n directions and 3 wave polarizations.
        power flow angles : np.ndarray of shape (n, 3)
            Power flow angles in radians.

    Notes
    -----
    The group velocity is calculated using the relation:
    v_g = (d/dk) * v_p / (2 * v_p)
    where v_g is the group velocity, d/dk is the derivative with respect to the wave vector,
    and v_p is the phase velocity.
    """

    # Ensure the input arrays have the correct shapes
    # assert phase_velocities.ndim == 2 and phase_velocities.shape[1] == 3, "phase_velocities must be of shape (n, 3)"
    # assert eigenvalue_gradients.ndim == 3 and eigenvalue_gradients.shape[1:] == (3, 3), "eigenvalue_gradients must be of shape (n, 3, 3)"

    # Calculate group velocity matrices
    phase_velocities_reshaped = phase_velocities[:, :, np.newaxis]
    group_velocities = eigenvalue_gradients / (2 * phase_velocities_reshaped)

    # Calculate magnitudes of group velocities
    group_velocity_magnitudes = np.linalg.norm(group_velocities, axis=2)

    # Calculate directions of group velocities (unit vectors)
    epsilon = 1e-10  # Add a small epsilon to avoid division by zero
    group_velocity_directions = group_velocities / (
        group_velocity_magnitudes[:, :, np.newaxis] + epsilon
    )

    # calculate thepower flow angles (in radians) for each wave direction
    # cos_power_flow_angles = np.dot(group_velocity_directions, wave_vectors.T)
    # power_flow_angles = np.arccos(np.clip(cos_power_flow_angles, -1, 1))

    return (
        group_velocities,
        group_velocity_magnitudes,
        group_velocity_directions,
        # power_flow_angles
    )


def calc_spherical_angles(vectors: np.ndarray) -> np.ndarray:
    """
    Convert unit vectors to spherical angles (theta, phi)
    in degrees.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, 3, 3)
        Direction vectors (need not be perfectly normalized).

    Returns
    -------
    theta_deg, phi_deg : np.ndarray
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
) -> dict:
    """
    Calculate group velocity vectors, magnitudes, directions, and power flow angles.

    Matches Jaeken & Cottenier (2016) implementation logic.

    Parameters
    ----------
    phase_velocities : (n, 3)
        [Vs2, Vs1, Vp] phase velocities (km/s).
    eigenvectors : (n, 3, 3)
        eigenvectors[n, mode, cart].
    christoffel_gradient : (n, 3, 3, 3)
        gradM[n, a, i, j] = ∂M_ij/∂q_a
    wave_vectors : (n, 3)
        Unit phase directions q.

    Returns
    -------
    dict with keys:
        grad_eigenvalues : (n, 3, 3)
        group_velocity   : (n, 3, 3)   (mode, cart)
        group_speed      : (n, 3)
        group_dir        : (n, 3, 3)
        group_theta_deg  : (n, 3)
        group_phi_deg    : (n, 3)
        cos_powerflow    : (n, 3)
        powerflow_deg    : (n, 3)
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
    cos_pf = np.einsum("nmi,ni->nm", group_dir, wave_vectors)
    cos_pf = np.clip(cos_pf, -1.0, 1.0)
    pf_rad = np.arccos(np.around(cos_pf, 10))
    pf_deg = np.rad2deg(pf_rad)

    # group direction spherical angles
    gtheta_deg, gphi_deg = calc_spherical_angles(group_dir)

    return {
        "grad_eigenvalues": grad_lam,
        "group_velocity": group_vel,
        "group_speed": group_speed,
        "group_dir": group_dir,
        "group_theta_deg": gtheta_deg,
        "group_phi_deg": gphi_deg,
        "cos_powerflow": cos_pf,
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
        NumPy array of shape (3, 3, 3, 3)
        with indices (i,j,k,l)

    Returns
    -------
    numpy.ndarray
        The Hessian of the Christoffel matrix
        (3, 3, 3, 3), hessM[a,b,i,j]
        

    Notes
    -----
    The function iterates over all combinations of the indices
    i, j, k, and L to compute the corresponding elements of the
    Hessian matrix.
    """

    # initialize array (pre-allocate)
    hessM = np.zeros((3, 3, 3, 3))

    for a in range(3):
        for b in range(3):
            hessM[a, b, :, :] = Cijkl[:, a, b, :] + Cijkl[:, b, a, :]

    return hessM


def _get_hessian_eigen(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    delta_Mij: np.ndarray,
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
        which has a shape of (n, 3)

    eigenvectors : numpy.ndarray
        The eigenvectors of the normalized Christoffel matrices,
        which has a shape of (n, 3, 3), rows=modes

    delta_Mij : numpy.ndarray
        The derivatives of the Christoffel matrices, which has a
        shape of (n, 3, 3, 3), [n, a, i, j]

    hess_matrix : numpy.ndarray
        The Hessian matrix, which has a shape of (3, 3, 3, 3)
        [a, b, i, j]
    
    tol : float
        Degeneracy tolerance for eigenvalue differences.

    Returns
    -------
    Hlam : (n, 3, 3, 3)
        Hlam[n, mode, a, b] = ∂²λ_mode / ∂q_a ∂q_b
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
    deriv = np.einsum("naij,nmj->nmai", delta_Mij, eigenvectors)

    # Term 2: 2 * deriv * pinv * deriv^T  -> (n, mode, a, b)
    term2 = 2.0 * np.einsum("nmai,nmij,nmbj->nmab", deriv, pinv, deriv)

    return term1 + term2


def calc_enhancement_factor(
    hessian_eigenvalues: np.ndarray,
    phase_velocities: np.ndarray,
    group_velocity: np.ndarray,
    group_speed: np.ndarray,
    wave_vectors: np.ndarray,
) -> np.ndarray:
    """
    Enhancement factor (exact, analytical)
    as in Jaeken & Cottenier (2016).

    Parameters
    ----------
    hessian_eigenvalues : (n, 3, 3, 3)
        Hlam[n, mode, a, b]
    phase_velocities : (n, 3)
    group_velocity : (n, 3, 3)
        vg[n, mode, cart]
    group_speed : (n, 3)
        |vg|
    wave_vectors : (n, 3)
        q

    Returns
    -------
    enhancement : (n, 3)
    """
    H = hessian_eigenvalues
    vp = phase_velocities
    vg = group_velocity
    vg_abs = group_speed
    q = wave_vectors

    # grad_group = d(vg)/dq from paper’s algebra (implemented in reference code)
    Hv = np.einsum("nmab,nmb->nma", H, vg)              # (n, mode, a)
    outer = vg[..., :, None] * Hv[..., None, :]         # (n, mode, a, b)

    grad_group = H / vg_abs[..., None, None] - outer / (vg_abs[..., None, None] ** 3)
    grad_group = grad_group / (2.0 * vp[..., None, None])

    cof = _cofactor_3x3(grad_group)                     # (n, mode, 3, 3)
    vec = np.einsum("nmij,nj->nmi", cof, q)             # (n, mode, 3)

    denom = np.linalg.norm(vec, axis=-1)
    return np.where(denom > 0.0, 1.0 / denom, np.inf)


def calc_power_flow_angles(group_dir: np.ndarray, wave_vectors: np.ndarray) -> np.ndarray:
    """
    Power flow angle (degrees) between group direction
    and phase direction.

    Parameters
    ----------
    group_dir : (n, 3, 3)
        Unit group directions, group_dir[n, mode, cart]
    wave_vectors : (n, 3)
        Unit phase directions

    Returns
    -------
    angles_deg : (n, 3)
    """
    cosang = np.einsum("nmi,ni->nm", group_dir, wave_vectors)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.rad2deg(np.arccos(np.around(cosang, 10)))


############################################################################################
def validate_cijs(Cij: np.ndarray) -> bool:
    """Input validation for 6x6 Voigt stiffness matrix."""
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij should be a 6x6 NumPy array.")
    if not np.allclose(Cij, Cij.T):
        raise ValueError("Cij should be symmetric.")
    return True


def validate_wavevectors(wavevectors) -> bool:
    """Input validation

    Parameters
    ----------
    wavevectors : np.ndarray of shape (3,) or (n, 3)
        The input array to validate.

    Returns
    -------
    bool : True if the array is of the correct shape and type,
        otherwise raises a ValueError.
    """
    if not isinstance(wavevectors, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if wavevectors.ndim not in [1, 2]:
        raise ValueError("Input array must be 1-dimensional or 2-dimensional.")

    if wavevectors.ndim == 1 and wavevectors.shape != (3,):
        raise ValueError("1-dimensional array must have shape (3,).")

    if wavevectors.ndim == 2 and wavevectors.shape[1] != 3:
        raise ValueError("2-dimensional array must have shape (n, 3).")

    return True


def _cofactor_3x3(A: np.ndarray) -> np.ndarray:
    """
    Vectorized cofactor for 3x3 matrices.

    Parameters
    ----------
    A : np.ndarray (n, 3, 3)

    Returns
    -------
    np.ndarray (n, 3, 3)
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
