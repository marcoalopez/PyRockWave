# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: layered_media.py                                          #
# Description: Module to calculate seismic reflectivity in layered    #
# media.                                                              #
#                                                                     #
# Copyright (c) 2023                                                  #
#                                                                     #
# PyRockWave is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published   #
# by the Free Software Foundation, either version 3 of the License,   #
# or (at your option) any later version.                              #
#                                                                     #
# PyRockWave is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of      #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
# GNU General Public License for more details.                        #
#                                                                     #
# You should have received a copy of the GNU General Public License   #
# along with PyRockWave. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                     #
# Author:                                                             #
# Marco A. Lopez-Sanchez, http://orcid.org/0000-0002-0261-9267        #
# Email: lopezmarco [to be found at] uniovi.es                        #
# Website: https://marcoalopez.github.io/PyRockWave/                  #
# Project Repository: https://github.com/marcoalopez/PyRockWave       #
#######################################################################

# Import statements
import numpy as np


# Function definitions
def snell(vp1, vp2, vs1, vs2, theta1):
    """
    Calculates the angles of refraction and reflection for an incident
    P-wave in a two-layered system.

    Parameters
    ----------
    vp1 : float or array-like
        P-wave velocity of upper layer.
    vs1 : float or array-like
        S-wave velocity of upper layer.
    vp2 : float or array-like
        P-wave velocity of lower layer.
    vs2 : float or array-like
        S-wave velocity of lower layer.
    theta1 : float or array-like
        Angle of incidence of P-wave in upper layer in degrees.

    Returns
    -------
    theta2 : float or array-like
        Angle of refraction for P-wave in lower layer in degrees.
    phi1 : float or array-like
        Angle of reflection for S-wave in upper layer in degrees.
    phi2 : float or array-like
        Angle of refraction for S-wave in lower layer in degrees.
    p : float or array-like
        Ray parameter.

    """
    
    # Convert angles to radians
    theta1 = np.deg2rad(theta1)
    
    # Calculate ray parameter using Snell's law
    p = np.sin(theta1) / vp1
    
    # Calculate reflection and refraction angles using Snell's law
    phi1 = np.arcsin(p * vs1)
    theta2 = np.arcsin(p * vp2)
    phi2 = np.arcsin(p * vs2)

    return theta2, phi1, phi2, p


# TODO
def calc_reflectivity(
    cij_upper: np.ndarray,
    cij_lower: np.ndarray,
    density_upper: float,
    density_lower: float,
    incident_angles
):
    pass


def reflectivity(a1, b1, e11, d11, e12, d12, g1, rho1,
                 a2, b2, e21, d21, e22, d22, g2, rho2,
                 theta):
    """
    Calculates the P-wave reflectivity in the symmetry plane for interfaces
    between 2 orthorhombic media.
    
    This is Python reimplementation with improved readability and
    documentation from srb toolbox (see Rorsym.m file) originally
    written by Diana Sava. 
    
    Parameters
    ----------
    a1, a2 : float or array-like
        P-wave vertical velocities of upper (1) and lower (2) medium.
    b1, b2 : float or array-like
        S-wave vertical velocities of upper (1) and lower (2) medium.
    e11, e21 : float or array-like
        Epsilon in the first symmetry plane of the orthorhombic
        medium for the upper (11) and lower (21) medium.
    d11, d21 : float or array-like
        Delta in the first symmetry plane of the orthorhombic
        medium for the upper (11) and lower (21) medium.
    e12, e22 : float or array-like
        Epsilon in the second symmetry plane of the orthorhombic
        medium for the upper (12) and lower (22) medium.
    d12, d22 : float or array-like
        Delta in the second symmetry plane of the orthorhombic
        medium for the upper (12) and lower (22) medium.
    g1, g2 : float or array-like
        Vertical shear wave splitting parameter for the
        upper (1) and lower (2) medium.
    rho1, rho2 : float or array-like
        Density of the upper (1) and lower (2) medium.
    theta : float or array-like
        Incident angle.

    Returns
    -------
    tuple of array-like
        Rxy: PP reflectivity as a function of angle of incidence
        in xz plane.
        Ryz: PP reflectivity as a function of angle of incidence
        in yz plane.

    Reference
    ---------
    Ruger, A., 1998, Variation of P-wave reflectivity coefficients
    with offset and azimuth in anisotropic media. Geophysics,
    Vol 63, No 3, p935. https://doi.org/10.1190/1.1444405
    """
    
    # Convert incident angle to radians
    theta_rad = np.deg2rad(theta)

    # Calculate impedance for both media
    Z1 = rho1 * a1
    Z2 = rho2 * a2

    # Calculate shear modulus for both media and their
    # average and difference
    G1 = rho1 * b1**2
    G2 = rho2 * b2**2
    G_avg = (G1 + G2) / 2.0
    G_diff = G2 - G1

    # Calculate average and difference for P-wave and
    # S-wave velocities 
    a_avg = (a1 + a2) / 2.0
    a_diff = a2 - a1
    b_avg = (b1 + b2) / 2.0

    # Calculate sin^2(theta) and tan^2(theta)
    sin_sq_theta = np.sin(theta_rad)**2
    tan_sq_theta = np.tan(theta_rad)**2

    # Calculate factor f 
    f = (2 * b_avg / a_avg)**2

    # Calculate reflectivity in xz plane (Rxz)
    Rxz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * (G_diff / G_avg - 2 * (g2 - g1)) + d22 - d12) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e22 - e12) * sin_sq_theta * tan_sq_theta)

    # Calculate reflectivity in yz plane (Ryz)
    Ryz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * G_diff / G_avg + d21 - d11) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e21 - e11) * sin_sq_theta * tan_sq_theta)

    return Rxz, Ryz


def Tsvankin_params(cij: np.ndarray, density: float):
    """Estimate the Tsvankin parameters for weak
    azimuthal orthotropic anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material
        in GPa.
    density : float
        The density of the material in g/cm3.

    Returns
    -------
    [float, float, float, float, float, float, float, float, float]
        List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, c22, c33, c44, c55, c66 = np.diag(cij)
    c12, c13, c23 = cij[0, 1], cij[0, 2],  cij[1, 2]

    # estimate the vertically propagating speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c55 / density)

    # estimate Tsvankin dimensionless parameters
    # VTI parameters in the YZ plane
    epsilon1 = (c22 - c33) / (2 * c33)
    delta1 = ((c23 + c44)**2 - (c33 - c44)**2) / (2*c33 * (c33 - c44))
    gamma1 = (c66 - c55) / (2 * c55)
    # VTI parameters in the XZ plane
    epsilon2 = (c11 - c33) / (2 * c33)
    delta2 = ((c13 + c55)**2 - (c33 - c55)**2) / (2*c33 * (c33 - c55))
    gamma2 = (c66 - c44) / (2 * c44)
    # VTI parameter in the XY plane
    delta3 = (c12 + c66)**2 - (c11 - c66)**2 / (2*c11 * (c11 - c66))

    return Vp0, Vs0, epsilon1, delta1, gamma1, epsilon2, delta2, gamma2, delta3


def schoenberg_muir_layered_medium(cij_layer1: np.ndarray,
                                   cij_layer2: np.ndarray,
                                   vfrac1: float,
                                   vfrac2: float):
    """Calculate the effective stiffness and compliance tensors for
    a layered medium using the Schoenberg & Muir approach.This
    function computes the effective stiffness and compliance
    tensors for a medium composed of two alternating thin layers.

    Parameters
    ----------
    cij_layer1 : numpy.ndarray
        6x6 stiffness tensor of the first layer
    cij_layer2 : numpy.ndarray
        6x6 stiffness tensor of the second layer
    vfrac1 : float between 0 and 1
        Volume (thickness) fraction of the first layer (0 <= vfrac1 <= 1)
    vfrac2 : float between 0 and 1
        Volume (thickness) fraction of the first layer (0 <= vfrac2 <= 1)

    Returns
    -------
    - effective_stiffness (numpy.ndarray): 6x6 effective stiffness tensor.
    - effective_compliance (numpy.ndarray): 6x6 effective compliance tensor.
    - effective_stiffness_from_compliance (numpy.ndarray): 6x6 effective
    stiffness tensor computed from the effective compliance
    
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

    # Input Error handling
    if not isinstance(cij_layer1, np.ndarray) or not isinstance(cij_layer2, np.ndarray):
        raise TypeError("cij_layer1 and cij_layer2 must be numpy arrays.")
    
    if cij_layer1.shape != (6, 6) or cij_layer2.shape != (6, 6):
        raise ValueError("cij_layer1 and cij_layer2 must be 6x6 arrays.")
    
    if not np.allclose(cij_layer1, cij_layer1.T):
        raise Exception("the elastic tensor 1 is not symmetric!")

    if not np.allclose(cij_layer2, cij_layer2.T):
        raise Exception("the elastic tensor 2 is not symmetric!")

    if not isinstance(vfrac1, (int, float)) or not isinstance(vfrac2, (int, float)):
        raise TypeError("vfrac1 and vfrac2 must be numbers.")

    if not (0 <= vfrac1 <= 1) or not (0 <= vfrac2 <= 1):
        raise ValueError("Volume fractions must be between 0 and 1.")

    if not np.isclose(vfrac1 + vfrac2, 1, rtol=1e-03):
        raise ValueError("Volume fractions must sum (approximately) to 1.")

    # Extract submatrices (partition) for Cnn, Ctn, Ctt from the stiffness tensor
    # normal
    Cnn1 = cij_layer1[2:5, 2:5]
    Cnn2 = cij_layer2[2:5, 2:5]
    # tangential-normal
    Ctn1 = cij_layer1[[0,1,5], 2:5]
    Ctn2 = cij_layer2[[0,1,5], 2:5]
    # tangential -> np.ix_(row_indices, col_indices)
    Ctt1 = cij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Ctt2 = cij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]
 
    # Compute effective normal stiffness
    Cnn = np.linalg.inv(vfrac1 * np.linalg.inv(Cnn1) + vfrac2 * np.linalg.inv(Cnn2))

    # Compute effective tangential-normal stiffness
    Ctn = (
        vfrac1 * Ctn1 @ np.linalg.inv(Cnn1) + vfrac2 * Ctn2 @ np.linalg.inv(Cnn2)
    ) @ Cnn

    # Compute effective tangential stiffness
    term1 = vfrac1 * Ctt1 + vfrac2 * Ctt2
    term2 = (vfrac1 * Ctn1 @ np.linalg.inv(Cnn1) @ Ctn1.T +
             vfrac2 * Ctn2 @ np.linalg.inv(Cnn2) @ Ctn2.T)
    term3 = (vfrac1 * Ctn1 @ np.linalg.inv(Cnn1) +
             vfrac2 * Ctn2 @ np.linalg.inv(Cnn2)) @ Cnn @ (
                    vfrac1 * np.linalg.inv(Cnn1) @ Ctn1.T +
                    vfrac2 * np.linalg.inv(Cnn2) @ Ctn2.T)

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

    # Extract submatrices (partition) for Snn, Stn, Snt, Stt from the compliance tensor
    # normal
    Snn1 = sij_layer1[2:5, 2:5]
    Snn2 = sij_layer2[2:5, 2:5]
    # tangential-normal
    Stn1 = sij_layer1[[0,1,5], 2:5]
    Stn2 = sij_layer2[[0,1,5], 2:5]
    # normal-tangential
    Snt1 = Stn1.T
    Snt2 = Stn2.T
    # tangential -> np.ix_(row_indices, col_indices)
    Stt1 = sij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Stt2 = sij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]

    # Compute effective normal compliance
    Snn = ((vfrac1 * Snn1 + vfrac2 * Snn2)
          - (vfrac1 * Snt1 @ np.linalg.inv(Stt1) @ Stn1 + vfrac2 * Snt2 @ np.linalg.inv(Stt2) @ Stn2)
          + (vfrac1 * Snt1 @ np.linalg.inv(Stt1) + vfrac2 * Snt2 @ np.linalg.inv(Stt2))
          @ np.linalg.inv(vfrac1 * np.linalg.inv(Stt1) + vfrac2 * np.linalg.inv(Stt2))
          @ (vfrac1 * np.linalg.inv(Stt1) @ Stn1 + vfrac2 * np.linalg.inv(Stt2) @ Stn2))

    # Compute effective tangential-normal compliance
    Stn = (
        np.linalg.inv(vfrac1 * np.linalg.inv(Stt1) + vfrac2 * np.linalg.inv(Stt2))
    ) @ (vfrac1 * np.linalg.inv(Stt2) @ Stn2 + vfrac2 * np.linalg.inv(Stt2) @ Stn2)

    # Compute effective tangential compliance
    Stt = np.linalg.inv(vfrac1 * np.linalg.inv(Stt1) + vfrac2 * np.linalg.inv(Stt2))

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


# End of file
