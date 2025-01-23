# -*- coding: utf-8 -*-
###############################################################################
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: orthotropic_models.py                                             #
# Description: This module calculates TODO                                    #
#                                                                             #
# Copyright (c) 2023-Present                                                  #
#                                                                             #
# License:                                                                    #
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
# Author:                                                                     #
# Marco A. Lopez-Sanchez                                                      #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Email: lopezmarco [to be found at] uniovi.es                                #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
###############################################################################

# Import statements
import numpy as np
import pandas as pd


# Function definitions
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


def HaoStovas_params(cij: np.ndarray, density: float):
    """Estimate the Hao and Stovas (2016) parameters modified
    from Alkhalifah (2003) for azimuthal orthotropic anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material
        in GPa.
    density : float
        The density of the material in g/cm3.

    Returns
    -------
    [float, float, float, float, float, float]
        TList containing the parameters
    """

    # unpack some elastic constants for readibility
    c11, c22, c33, c44, c55, c66 = np.diag(cij)
    c12, c13, c23 = cij[0, 1], cij[0, 2],  cij[1, 2]

    # estimate the vertically propagating P-speed
    Vp0 = np.sqrt(c33 / density)

    # estimate Hao-Stovas-Alkhalifah dimensionless parameters
    epsilon1 = np.sqrt((c22*(c33 - c44)) / (c23**2 + 2*c23*c44 + c33*c44))
    epsilon2 = np.sqrt((c11*(c33 - c55)) / (c13**2 + 2*c13*c55 + c33*c55))
    epsilon3 = np.sqrt((c22*(c11 - c66)) / (c12**2 + 2*c12*c66 + c11*c66))
    r1 = (c23**2 + 2*c23*c44 + c33*c44) / (c33*(c33 - c44))
    r2 = (c13**2 + 2*c13*c55 + c33*c55) / (c33*(c33 - c55))

    return Vp0, epsilon1, epsilon2, epsilon3, r1, r2


def orthotropic_azimuthal_anisotropy(elastic, wavevectors):
    """The simplest realistic case of azimuthal anisotropy is that of
    orthorhombic anisotropy (a.k.a. orthotropic).
    See https://doi.org/10.3389/feart.2023.1261033

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

    # get Hao and Stovas parameters
    Vp0, ε1, ε2, ε3, r1, r2 = HaoStovas_params(elastic.Cij, elastic.density)

    # estimate α(φ) and β(φ)
    alphaPhi = _calc_alphaPhi(azimuths, ε1, ε2, ε3, r1, r2)
    betaPhi = r1 * np.sin(azimuths)**2 + r2 * np.cos(azimuths)**2 - alphaPhi

    # estimate phase wavespeeds as a function of propagation
    # polar (θ) and azimuth (φ) angles
    Vp = np.sqrt(
        0.5 * Vp0**2 * (np.cos(polar)**2 + alphaPhi * np.sin(polar)**2) +
        0.5 * Vp0**2 * np.sqrt((np.cos(polar)**2 + alphaPhi * np.sin(polar))**2 +
                               (4 * betaPhi * np.cos(polar)**2 * np.sin(polar**2)))
    )

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp
            }

    return pd.DataFrame(data)


def _calc_alphaPhi(azimuths, ε1, ε2, ε3, r1, r2):
    first_term = 0.5 * (
        (r2 * ε2**2 * np.cos(azimuths) ** 2) +
        (r1 * ε1**2 * np.sin(azimuths) ** 2)
    )
    
    a = (r2 * ε2**2 * np.cos(azimuths) ** 2) + \
        (r1 * ε1**2 * np.sin(azimuths) ** 2)
    b = (1 / ε3**2) * r1 * r2 * ε1**2 * ε2**2 * np.sin(2 * azimuths) ** 2
    second_term = 0.5 * np.sqrt(a**2 + b)

    return first_term + second_term


# End of file
