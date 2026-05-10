# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: layered_media.py                                                  #
# Description: This module calculates seismic reflectivity in layered media.  #
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
# Email: lopezmarco [to be found at] uniovi dot es                            #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
# =========================================================================== #

# Import statements
import numpy as np


# Function definitions
def snell(
    vp_upper_layer: float | np.ndarray,
    vp_lower_layer: float | np.ndarray,
    vs_upper_layer: float | np.ndarray,
    vs_lower_layer: float | np.ndarray,
    theta1: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the angles of refraction and reflection for an incident
    P-wave in a two-layered system.

    Parameters
    ----------
    vp_upper_layer : float or array-like
        P-wave velocity of upper layer.
    vp_lower_layer : float or array-like
        P-wave velocity of lower layer.
    vs_upper_layer : float or array-like
        S-wave velocity of upper layer.
    vs_lower_layer : float or array-like
        S-wave velocity of lower layer.
    theta1 : float or array-like
        Angle(s) of incidence of P-wave in upper layer in degrees.
        Must be in [0, 90).

    Returns
    -------
    theta2 : float or array-like
        Angle(s) of refraction for P-wave in lower layer in degrees.
    phi1 : float or array-like
        Angle(s) of reflection for S-wave in upper layer in degrees.
    phi2 : float or array-like
        Angle(s) of refraction for S-wave in lower layer in degrees.
    p : float or array-like
        Ray parameter(s).

    Notes
    -----
    At and beyond the critical angle np.arcsin returns nan silently;
    no exception is raised.
    """

    if (np.any(np.asarray(vp_upper_layer) <= 0) or np.any(np.asarray(vp_lower_layer) <= 0)
            or np.any(np.asarray(vs_upper_layer) <= 0) or np.any(np.asarray(vs_lower_layer) <= 0)):
        raise ValueError("All velocities must be positive.")
    if np.any(np.asarray(theta1) < 0) or np.any(np.asarray(theta1) >= 90):
        raise ValueError("theta1 must be in [0, 90).")

    # Convert angles to radians
    theta1 = np.deg2rad(theta1)

    # Calculate ray parameter using Snell's law
    p = np.sin(theta1) / vp_upper_layer

    # Calculate reflection and refraction angles using Snell's law
    phi1 = np.arcsin(p * vs_upper_layer)
    theta2 = np.arcsin(p * vp_lower_layer)
    phi2 = np.arcsin(p * vs_lower_layer)

    return theta2, phi1, phi2, p


def calc_reflectivity(
    cij_upper_layer: np.ndarray,
    cij_lower_layer: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    incident_angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that computes PP reflectivity in both symmetry
    planes for an interface between two orthorhombic media.

    Extracts Tsvankin parameters from the input stiffness tensors and
    calls :func:`reflectivity`.

    Parameters
    ----------
    cij_upper_layer : np.ndarray
        The 6x6 elastic stiffness tensor of the upper layer in GPa.
    cij_lower_layer : np.ndarray
        The 6x6 elastic stiffness tensor of the lower layer in GPa.
    upper_density_gcm3 : float
        The density of the upper layer in g/cm³.
    lower_density_gcm3 : float
        The density of the lower layer in g/cm³.
    incident_angles_deg : np.ndarray
        Incident angles in degrees.

    Returns
    -------
    Rxz : np.ndarray
        PP reflectivity as a function of angle of incidence in the xz plane.
    Ryz : np.ndarray
        PP reflectivity as a function of angle of incidence in the yz plane.
    """

    # Compute Tsvankin params for both layers
    upper_layer_params = tsvankin_params(cij_upper_layer, upper_density_gcm3)
    lower_layer_params = tsvankin_params(cij_lower_layer, lower_density_gcm3)

    # Delegate to reflectivity, converting angles from degrees to radians
    return reflectivity(
        upper_layer_params,
        lower_layer_params,
        upper_density_gcm3,
        lower_density_gcm3,
        np.deg2rad(incident_angles_deg),
    )


def reflectivity(
    upper_layer_tsvankin_params: np.ndarray,
    lower_layer_tsvankin_params: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    incident_angles_rad: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the P-wave reflectivity in the symmetry plane for interfaces
    between 2 orthorhombic media using the approach of Ruger (1998).

    This is Python reimplementation with improved readability and
    documentation from srb toolbox (https://github.com/StanfordRockPhysics/SRBToolbox)
    In particular, a reimplementation of the  Rorsym.m file) originally
    written by Diana Sava

    Parameters
    ----------
    upper_layer_tsvankin_params : array-like of shape (9,)
        The Tsvankin params of the upper medium (output of
        :func:`tsvankin_params`).
    lower_layer_tsvankin_params : array-like of shape (9,)
        The Tsvankin params of the lower medium (output of
        :func:`tsvankin_params`).
    upper_density_gcm3 : float
        Density of the upper medium in g/cm³.
    lower_density_gcm3 : float
        Density of the lower medium in g/cm³.
    incident_angles_rad : float or array-like
        Incident angle(s) in radians.

    Arrays with Tsvankin params include the following params in this
    specific order (for details refer to :func:`tsvankin_params`):
    -----------------------------------------------------------------
    Vp0 : P-wave vertical velocity of medium (km/s).
    Vs0 : S-wave vertical velocity of medium (km/s).
    epsilon1 (e1) : Epsilon in the first symmetry plane of the
        orthorhombic medium (normal to x).
    delta1 (d1) : Delta in the first symmetry plane of the
        orthorhombic medium (normal to x).
    epsilon2 (e2) : Epsilon in the second symmetry plane of the
        orthorhombic medium (normal to y).
    delta2 (d2) : Delta in the second symmetry plane of the
        orthorhombic medium (normal to y).
    gamma1 (g1) : Vertical shear wave splitting parameter of the medium.
    gamma2, delta3 : Present in the array but not used in this
        reflectivity calculation.

    Returns
    -------
    Rxz : np.ndarray
        PP reflectivity as a function of angle of incidence in the xz plane.
    Ryz : np.ndarray
        PP reflectivity as a function of angle of incidence in the yz plane.

    Reference
    ---------
    Ruger, A., 1998, Variation of P-wave reflectivity coefficients
    with offset and azimuth in anisotropic media. Geophysics,
    Vol 63, No 3, p935. https://doi.org/10.1190/1.1444405
    """

    if (len(upper_layer_tsvankin_params) < 7
            or len(lower_layer_tsvankin_params) < 7):
        raise ValueError("Tsvankin parameter arrays must have at least 7 elements.")
    if upper_density_gcm3 <= 0 or lower_density_gcm3 <= 0:
        raise ValueError("Densities must be positive.")

    # Extract Tsvankin params
    # Order: Vp0, Vs0, epsilon1, delta1, epsilon2, delta2, gamma1, gamma2, delta3
    Vp0_up, Vs0_up, e1_up, d1_up, e2_up, d2_up, g1_up, *_ = upper_layer_tsvankin_params
    Vp0_low, Vs0_low, e1_low, d1_low, e2_low, d2_low, g1_low, *_ = lower_layer_tsvankin_params

    # Calculate impedance for both media
    Z1 = upper_density_gcm3 * Vp0_up
    Z2 = lower_density_gcm3 * Vp0_low

    # Calculate shear modulus for both media and their average and difference
    G1 = upper_density_gcm3 * Vs0_up**2
    G2 = lower_density_gcm3 * Vs0_low**2
    G_avg = (G1 + G2) / 2.0
    G_diff = G2 - G1

    # Calculate average and difference for P-wave and S-wave velocities
    a_avg = (Vp0_up + Vp0_low) / 2.0
    a_diff = Vp0_low - Vp0_up
    b_avg = (Vs0_up + Vs0_low) / 2.0

    # Calculate sin²(θ) and tan²(θ)
    sin_sq_theta = np.sin(incident_angles_rad)**2
    tan_sq_theta = np.tan(incident_angles_rad)**2

    # Calculate factor f
    f = (2 * b_avg / a_avg)**2

    # Calculate reflectivity in xz plane (Rxz) — Ruger (1998) Eq. 7
    Rxz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * (G_diff / G_avg + 2 * (g1_low - g1_up)) + d2_low - d2_up) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e2_low - e2_up) * sin_sq_theta * tan_sq_theta)

    # Calculate reflectivity in yz plane (Ryz) — Ruger (1998) Eq. 7
    Ryz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * G_diff / G_avg + d1_low - d1_up) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e1_low - e1_up) * sin_sq_theta * tan_sq_theta)

    return Rxz, Ryz


def tsvankin_params(
    cij: np.ndarray,
    density: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Estimate the Tsvankin parameters for weak
    azimuthal orthotropic anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The 6x6 elastic stiffness tensor of the material in GPa.
    density : float
        The density of the material in g/cm³.

    Returns
    -------
    Vp0 : float
        P-wave vertical velocity in km/s.
    Vs0 : float
        S-wave vertical velocity in km/s.
    epsilon1 : float
        Epsilon in the first symmetry plane (normal to x).
    delta1 : float
        Delta in the first symmetry plane (normal to x).
    epsilon2 : float
        Epsilon in the second symmetry plane (normal to y).
    delta2 : float
        Delta in the second symmetry plane (normal to y).
    gamma1 : float
        Shear-wave splitting parameter for the yz plane.
    gamma2 : float
        Shear-wave splitting parameter for the xz plane.
    delta3 : float
        Delta in the xy symmetry plane.
    """
    if not isinstance(cij, np.ndarray) or cij.shape != (6, 6):
        raise ValueError("cij must be a 6x6 numpy array.")
    if not np.allclose(cij, cij.T):
        raise ValueError("cij must be symmetric.")
    if density <= 0:
        raise ValueError("density must be positive.")

    # Unpack some elastic constants for readability
    c11, c22, c33, c44, c55, c66 = np.diag(cij)
    c12, c13, c23 = cij[0, 1], cij[0, 2], cij[1, 2]

    # Estimate the vertically propagating speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c55 / density)

    # Estimate Tsvankin dimensionless parameters
    # VTI parameters in the YZ plane
    epsilon1 = (c22 - c33) / (2 * c33)
    delta1 = ((c23 + c44) ** 2 - (c33 - c44) ** 2) / (2 * c33 * (c33 - c44))
    gamma1 = (c66 - c55) / (2 * c55)
    # VTI parameters in the XZ plane
    epsilon2 = (c11 - c33) / (2 * c33)
    delta2 = ((c13 + c55) ** 2 - (c33 - c55) ** 2) / (2 * c33 * (c33 - c55))
    gamma2 = (c66 - c44) / (2 * c44)
    # VTI parameter in the XY plane
    delta3 = ((c12 + c66) ** 2 - (c11 - c66) ** 2) / (2 * c11 * (c11 - c66))

    return Vp0, Vs0, epsilon1, delta1, epsilon2, delta2, gamma1, gamma2, delta3


def schoenberg_muir_layered_medium(
    cij_layer1: np.ndarray,
    cij_layer2: np.ndarray,
    vfrac1: float,
    vfrac2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the effective stiffness and compliance tensors for
    a layered medium using the Schoenberg & Muir approach. This
    function computes the effective stiffness and compliance
    tensors for a medium composed of two alternating thin layers.

    Parameters
    ----------
    cij_layer1 : numpy.ndarray
        6x6 stiffness tensor of the first layer.
    cij_layer2 : numpy.ndarray
        6x6 stiffness tensor of the second layer.
    vfrac1 : float
        Volume (thickness) fraction of the first layer (0 <= vfrac1 <= 1).
    vfrac2 : float
        Volume (thickness) fraction of the second layer (0 <= vfrac2 <= 1).

    Returns
    -------
    effective_stiffness : numpy.ndarray of shape (6, 6)
        Effective stiffness tensor.
    effective_compliance : numpy.ndarray of shape (6, 6)
        Effective compliance tensor.
    effective_stiffness_from_compliance : numpy.ndarray of shape (6, 6)
        Effective stiffness tensor computed from the effective compliance.

    References
    ----------
    Schoenberg, M., & Muir, F. (1989). A calculus for finely layered
    anisotropic media. Geophysics, 54(5), 581-589.
    https://doi.org/10.1190/1.1442685

    Nichols, D., Muir, F., & Schoenberg, M. (1989). Elastic
    properties of rocks with multiple sets of fractures. SEG
    Technical Program Expanded Abstracts: 471-474.
    https://doi.org/10.1190/1.1889682

    Notes
    -----
    This is Python reimplementation with improved readability and
    documentation from srb toolbox (see sch_muir_bckus.m file)
    originally written by Kaushik Bandyopadhyay in 2008.
    """

    _validate_schoenberg_muir_layered_medium(cij_layer1, cij_layer2, vfrac1, vfrac2)

    # Extract submatrices (partition) for Cnn, Ctn, Ctt from the stiffness tensor
    # normal
    Cnn1 = cij_layer1[2:5, 2:5]
    Cnn2 = cij_layer2[2:5, 2:5]
    # tangential-normal
    Ctn1 = cij_layer1[[0, 1, 5], 2:5]
    Ctn2 = cij_layer2[[0, 1, 5], 2:5]
    # tangential -> np.ix_(row_indices, col_indices)
    Ctt1 = cij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Ctt2 = cij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]

    # Pre-compute inverses used multiple times
    Cnn1_inv = np.linalg.inv(Cnn1)
    Cnn2_inv = np.linalg.inv(Cnn2)

    # Compute effective normal stiffness
    Cnn = np.linalg.inv(vfrac1 * Cnn1_inv + vfrac2 * Cnn2_inv)

    # Compute effective tangential-normal stiffness
    Ctn = (vfrac1 * Ctn1 @ Cnn1_inv + vfrac2 * Ctn2 @ Cnn2_inv) @ Cnn

    # Compute effective tangential stiffness
    term1 = vfrac1 * Ctt1 + vfrac2 * Ctt2
    term2 = vfrac1 * Ctn1 @ Cnn1_inv @ Ctn1.T + vfrac2 * Ctn2 @ Cnn2_inv @ Ctn2.T
    term3 = (vfrac1 * Ctn1 @ Cnn1_inv + vfrac2 * Ctn2 @ Cnn2_inv) @ Cnn @ (
                vfrac1 * Cnn1_inv @ Ctn1.T + vfrac2 * Cnn2_inv @ Ctn2.T)

    Ctt = term1 - term2 + term3

    # Assemble effective stiffness tensor in full Voigt notation
    effective_stiffness = np.array(
        [
            [Ctt[0, 0], Ctt[0, 1], Ctn[0, 0], Ctn[0, 1], Ctn[0, 2], Ctt[0, 2]],
            [Ctt[1, 0], Ctt[1, 1], Ctn[1, 0], Ctn[1, 1], Ctn[1, 2], Ctt[1, 2]],
            [Ctn[0, 0], Ctn[1, 0], Cnn[0, 0], Cnn[1, 0], Cnn[0, 2], Ctn[2, 0]],
            [Ctn[0, 1], Ctn[1, 1], Cnn[1, 0], Cnn[1, 1], Cnn[1, 2], Ctn[2, 1]],
            [Ctn[0, 2], Ctn[1, 2], Cnn[0, 2], Cnn[1, 2], Cnn[2, 2], Ctn[2, 2]],
            [Ctt[0, 2], Ctt[1, 2], Ctn[2, 0], Ctn[2, 1], Ctn[2, 2], Ctt[2, 2]],
        ]
    )

    # Compute compliances from stiffnesses
    sij_layer1 = np.linalg.inv(cij_layer1)
    sij_layer2 = np.linalg.inv(cij_layer2)

    # Extract submatrices (partition) for Snn, Stn, Stt from the compliance tensor
    # normal
    Snn1 = sij_layer1[2:5, 2:5]
    Snn2 = sij_layer2[2:5, 2:5]
    # tangential-normal
    Stn1 = sij_layer1[[0, 1, 5], 2:5]
    Stn2 = sij_layer2[[0, 1, 5], 2:5]
    # tangential -> np.ix_(row_indices, col_indices)
    Stt1 = sij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Stt2 = sij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]

    # Pre-compute inverses used multiple times
    Stt1_inv = np.linalg.inv(Stt1)
    Stt2_inv = np.linalg.inv(Stt2)

    # Compute effective tangential compliance (needed by Stn and Snn below)
    Stt = np.linalg.inv(vfrac1 * Stt1_inv + vfrac2 * Stt2_inv)

    # Compute effective tangential-normal compliance
    Stn = Stt @ (vfrac1 * Stt1_inv @ Stn1 + vfrac2 * Stt2_inv @ Stn2)

    # Compute effective normal compliance
    Snn = ((vfrac1 * Snn1 + vfrac2 * Snn2)
           - (vfrac1 * Stn1.T @ Stt1_inv @ Stn1 + vfrac2 * Stn2.T @ Stt2_inv @ Stn2)
           + (vfrac1 * Stn1.T @ Stt1_inv + vfrac2 * Stn2.T @ Stt2_inv)
           @ Stt
           @ (vfrac1 * Stt1_inv @ Stn1 + vfrac2 * Stt2_inv @ Stn2))

    # Assemble effective compliance matrix
    effective_compliance = np.array(
        [
            [Stt[0, 0], Stt[0, 1], Stn[0, 0], Stn[0, 1], Stn[0, 2], Stt[0, 2]],
            [Stt[1, 0], Stt[1, 1], Stn[1, 0], Stn[1, 1], Stn[1, 2], Stt[1, 2]],
            [Stn[0, 0], Stn[1, 0], Snn[0, 0], Snn[1, 0], Snn[0, 2], Stn[2, 0]],
            [Stn[0, 1], Stn[1, 1], Snn[1, 0], Snn[1, 1], Snn[1, 2], Stn[2, 1]],
            [Stn[0, 2], Stn[1, 2], Snn[0, 2], Snn[1, 2], Snn[2, 2], Stn[2, 2]],
            [Stt[0, 2], Stt[1, 2], Stn[2, 0], Stn[2, 1], Stn[2, 2], Stt[2, 2]],
        ]
    )

    # Compute stiffness from effective compliance
    effective_stiffness_from_compliance = np.linalg.inv(effective_compliance)

    return effective_stiffness, effective_compliance, effective_stiffness_from_compliance


# =================================================================
# Private helpers for internal use only

def _validate_schoenberg_muir_layered_medium(
    cij_layer1: np.ndarray,
    cij_layer2: np.ndarray,
    vfrac1: float,
    vfrac2: float,
) -> None:
    """Validate inputs for :func:`schoenberg_muir_layered_medium`."""
    if not isinstance(cij_layer1, np.ndarray) or not isinstance(cij_layer2, np.ndarray):
        raise TypeError("cij_layer1 and cij_layer2 must be numpy arrays.")

    if cij_layer1.shape != (6, 6) or cij_layer2.shape != (6, 6):
        raise ValueError("cij_layer1 and cij_layer2 must be 6x6 arrays.")

    if not np.allclose(cij_layer1, cij_layer1.T):
        raise ValueError("the elastic tensor 1 is not symmetric!")

    if not np.allclose(cij_layer2, cij_layer2.T):
        raise ValueError("the elastic tensor 2 is not symmetric!")

    if not isinstance(vfrac1, (int, float)) or not isinstance(vfrac2, (int, float)):
        raise TypeError("vfrac1 and vfrac2 must be numbers.")

    if not (0 <= vfrac1 <= 1) or not (0 <= vfrac2 <= 1):
        raise ValueError("Volume fractions must be between 0 and 1.")

    if not np.isclose(vfrac1 + vfrac2, 1, rtol=1e-03):
        raise ValueError("Volume fractions must sum (approximately) to 1.")


# End of file
