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


# End of file
