# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: orthotropic_models.py                                     #
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
import pandas as pd


# Function definitions
def Tsvankin_params(cij: np.ndarray, density: float):
    """Estimate the Tsvankin paramenters for
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
    [float, float, (float, float, float, float, float, float, float)]
        Tuple/List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, c22, c33, c44, c55, c66 = np.diag(cij)
    c12, c13, c23 = cij[0, 1], cij[0, 2],  cij[1, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c55 / density)

    # estimate Tsvankin dimensionless parameters
    epsilon1 = (c22 - c33) / (2 * c33)
    delta1 = ((c23 + c44)**2 - (c33 - c44)**2) / (2*c33 * (c33 - c44))
    gamma1 = (c66 - c55) / (2 * c55)
    epsilon2 = (c11 - c33) / (2 * c33)
    delta2 = ((c13 + c55)**2 - (c33 - c55)**2) / (2*c33 * (c33 - c55))
    gamma2 = (c66 - c44) / (2 * c44)
    delta3 = (c12 + c66)**2 - (c11 - c66)**2 / (2*c11 * (c11 - c66))

    return Vp0, Vs0, (epsilon1, delta1, gamma1, epsilon2, delta2, gamma2, delta3)


def orthotropic_azimuthal_anisotropy(elastic, wavevectors):
    """The simplest realistic case of azimuthal anisotropy is that of
    orthorhombic anisotropy (a.k.a. orthotropic).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.

    wavevectors : numpy.ndarray
        The wave vectors in spherical coordinates

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using the orthorhombic
        anisotropy model.
    """
    # extract azimuths and polar angles
    azimuths, polar = wavevectors

    # get Tsvankin parameters
    Vp0, Vs0, *params = Tsvankin_params(elastic.Cij, elastic.density)
    epsilon1, delta1, gamma1, epsilon2, delta2, gamma2, delta3 = params

    # estimate phase wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = Vp0 * (1 + delta2 * sin_theta**2 * cos_theta**2 + epsilon2 * sin_theta**4)
    # TODO
    # Vsv = Vs0 * (1 + (Vp0 / Vs0)**2 * (epsilon - delta) * sin_theta**2 * cos_theta**2)
    # Vsh = Vs0 * (1 + gamma * sin_theta**2)

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp,
            'Vsv': Vsv,
            'Vsh': Vsh}

    return pd.DataFrame(data)

# End of file
