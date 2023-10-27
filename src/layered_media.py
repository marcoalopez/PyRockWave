# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: layered_media.py                                          #
# Description: TODO                                                   #
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


def calculate_reflectivity(a1, b1, e11, d11, e12, d12, g1, rho1,
                           a2, b2, e21, d21, e22, d22, g2, rho2,
                           theta):
    """
    Calculates the reflectivity in the symmetry plane for interfaces
    between 2 orthorhombic media. This is refactored code with improved
    readability and documentation from srb toolbox written by Diana Sava.
    
    Parameters
    ----------
    a1 : float or array-like
        P-wave vertical velocities of upper medium (1).
    b1 : float or array-like
        S-wave vertical velocities of upper medium (1).
    e11 : float or array-like
        Epsilon in the first symmetry plane of the orthorhombic
        medium for the upper medium.
    d11 : float or array-like
        Delta in the first symmetry plane of the orthorhombic
        medium for the upper medium.
    e12 : float or array-like
        Epsilon in the second symmetry plane of the orthorhombic
        medium for the upper medium.
    d12 : float or array-like
        Delta in the second symmetry plane of the orthorhombic
        medium for the upper medium.
    g1 : float or array-like
        Vertical shear wave splitting parameter for the
        upper medium (1).
    rho1 : float or array-like
        Density of the upper medium.
    a2 : float or array-like
        P-wave vertical velocities of lower medium (2).
    b2 : float or array-like
        S-wave vertical velocities of lower medium (2).
    e21 : float or array-like
        Epsilon in the first symmetry plane of the orthorhombic
        medium for the lower medium.
    d21 : float or array-like
        Delta in the first symmetry plane of the orthorhombic
        medium for the lower medium.
    e22 : float or array-like
        Epsilon in the second symmetry plane of the orthorhombic
        medium for the lower medium.
    d22 : float or array-like
        Delta in the second symmetry plane of the orthorhombic
        medium for the lower medium.
    g2 : float or array-like
        Vertical shear wave splitting parameter for the lower
        medium (2).
    rho2 : float or array-like
        Density of the lower medium.
    theta : float or array-like
        Incident angle.

    Returns
    -------
    tuple of array-like
        Rxy: PP reflectivity as a function of angle of incidence
        in xz plane (13).
        Ryz: PP reflectivity as a function of angle of incidence
        in yz plane (23).

    Reference
    ---------
    Ruger, A., 1998, Variation of P-wave reflectivity coefficients
    with offset and azimuth in anisotropic media. Geophysics,
    Vol 63, No 3, p935.
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

    # Calculate reflectivity in xz plane (Rxy)
    Rxz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * (G_diff / G_avg - 2 * (g2 - g1)) + d22 - d12) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e22 - e12) * sin_sq_theta * tan_sq_theta)

    # Calculate reflectivity in yz plane (Ryz)
    Ryz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * G_diff / G_avg + d21 - d11) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e21 - e11) * sin_sq_theta * tan_sq_theta)

    return Rxz, Ryz


# End of file
