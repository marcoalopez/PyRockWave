###############################################################################
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: anisotropic_models.py                                             #
# Description: This module calculates TODO                                    #
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
# Email: lopezmarco [to be found at] uniovi dot es                            #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
###############################################################################

# Import statements
import numpy as np
import pandas as pd


# Function definitions
def weak_polar_anisotropy(
    cij: np.ndarray,
    density_gcm3: float,
    wavevectors_rad: np.ndarray
):
    """Calculate the velocity of body waves in a material as a function
    of the direction of propagation, assuming that the elastic properties
    of the material have a weak polar anisotropy, using the Thomsen
    approach (Thomsen, 1986: https://doi.org/10.1190/1.1442051).

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic tensor of the material in 6x6 shape

    density : float
        The density of the material in g/cm3.

    wavevectors_rad : numpy.ndarray
        The wave vectors in spherical coordinates (radians)

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vsv, Vsh, Vs1, and Vs2 speeds and
        polarization anisotropy using the weak polar anisotropy
        model. Vsv & Vsh - S-waves with vertical and horizontal
        polarization respectively.
    """

    # extract azimuths and polar angles
    azimuths, polar = wavevectors_rad

    # compute Thomsen parameters
    Vp0, Vs0, epsilon, delta, gamma = Thomsen_params(cij, density_gcm3)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = Vp0 * (1 + delta * sin_theta**2 * cos_theta**2 + epsilon * sin_theta**4)
    Vsv = Vs0 * (1 + (Vp0 / Vs0) ** 2 * (epsilon - delta) * sin_theta**2 * cos_theta**2)
    Vsh = Vs0 * (1 + gamma * sin_theta**2)

    # calc Vs1 (fast) and Vs2 (slow) shear waves
    Vs1 = np.maximum(Vsv, Vsh)
    Vs2 = np.minimum(Vsv, Vsh)

    # estimate polarization anisotropy in percentage
    ShearWaveSplitting = 200 * (Vs1 - Vs2) / (Vs1 + Vs2)

    # reshape and store arrays
    data = {
        "polar_ang": polar,
        "azimuthal_ang": azimuths,
        "Vsv": Vsv,
        "Vsh": Vsh,
        "Vp": Vp,
        "Vs1": Vs1,
        "Vs2": Vs2,
        "SWS": np.around(ShearWaveSplitting, 1),
    }

    return pd.DataFrame(data)


def polar_anisotropy(
    cij: np.ndarray,
    density_gcm3: float,
    wavevectors_rad: np.ndarray
):
    """Calculate the velocity of body waves in a material as a function
    of the direction of propagation, assuming that the elastic properties
    of the material have polar anisotropy, using the Anderson approach
    (Anderson, 1961: https://doi.org/10.1029/JZ066i009p02953).

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic tensor of the material in 6x6 shape

    density : float
        The density of the material in g/cm3.

    wavevectors_rad : numpy.ndarray
        The wave vectors in spherical coordinates (radians)

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vsv, Vsh, Vs1, and Vs2 speeds and
        polarization anisotropy using the weak polar anisotropy
        model.
    """

    # extract azimuths and polar angles
    azimuths, polar = wavevectors_rad

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(cij)
    c13 = cij[0, 2]

    # estimate D value
    first_term = (c33 - c44) ** 2
    second_term = (
        2
        * (2 * (c13 + c44) ** 2 - (c33 - c44) * (c11 + c33 - 2 * c44))
        * np.sin(polar) ** 2
    )
    third_term = ((c11 + c33 - 2 * c44) ** 2 - 4 * (c13 + c44) ** 2) * np.sin(
        polar
    ) ** 4
    D = np.sqrt(first_term + second_term + third_term)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = np.sqrt(
        (1 / (2 * density_gcm3)) * (c33 + c44 + (c11 - c33) * sin_theta**2 + D)
    )
    Vsv = np.sqrt(
        (1 / (2 * density_gcm3)) * (c33 + c44 + (c11 - c33) * sin_theta**2 - D)
    )
    Vsh = np.sqrt((1 / density_gcm3) * (c44 * cos_theta**2 + c66 * sin_theta**2))
    
    # calc Vs1 (fast) and Vs2 (slow) shear waves
    Vs1 = np.maximum(Vsv, Vsh)
    Vs2 = np.minimum(Vsv, Vsh)

    # estimate polarization anisotropy in percentage
    ShearWaveSplitting = 200 * (Vs1 - Vs2) / (Vs1 + Vs2)

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vsv': Vsv,
            'Vsh': Vsh,
            'Vp': Vp,
            'Vs1': Vs1,
            'Vs2': Vs2,
            'SWS': np.around(ShearWaveSplitting, 1)}

    return pd.DataFrame(data)


def orthotropic_azimuthal_anisotropy(
    cij: np.ndarray,
    density_gcm3: float,
    wavevectors_rad: np.ndarray
):
    """Calculate the velocity of body P-waves in a material as a function
    of the direction of propagation, assuming that the elastic properties
    of the material have orthorhombic anisotropy (a.k.a. orthotropic),
    using the procedure described in Wang et al. (2023):
    See https://doi.org/10.3389/feart.2023.1261033

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic tensor of the material in 6x6 shape

    density : float
        The density of the material in g/cm3.

    wavevectors_rad : numpy.ndarray
        The wave vectors in spherical coordinates (radians)

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using the orthorhombic
        anisotropy model.
    """
    # extract azimuths and polar angles
    azimuths, polar = wavevectors_rad

    # get Hao and Stovas parameters
    Vp0, ε1, ε2, ε3, r1, r2 = HaoStovas_params(cij, density_gcm3)

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


###############################################################################
# functions that compute model parameters
def Thomsen_params(cij: np.ndarray, density_gcm3: float):
    """Estimate the Thomsen parameters for
    weak polar anisotropy.

    Thomsen parameters:
    - Vp0: P-wave velocity in vertical direction.
    - Vs0: V wave velocity in vertical direction.
    - ε (Epsilon): Describes the difference in P-wave velocity
    between the horizontal and vertical directions.
    - δ (Delta): Relates to the angular dependence of P-wave
    velocity near the vertical direction.
    - γ (Gamma): Describes the difference in S-wave velocity
    between the horizontal and vertical directions

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material
        in GPa.
    density : float
        The density of the material in g/cm3.

    Returns
    -------
    [float, float, float, float, float]
        Tuple/List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(cij)
    c13 = cij[0, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density_gcm3)
    Vs0 = np.sqrt(c44 / density_gcm3)

    # estimate Thomsen dimensionless parameters
    epsilon = (c11 - c33) / (2 * c33)
    delta = ((c13 + c44)**2 - (c33 - c44)**2) / (2 * c33 * (c33 - c44))
    gamma = (c66 - c44) / (2 * c44)

    return Vp0, Vs0, epsilon, delta, gamma


def Tsvankin_params(cij: np.ndarray, density_gcm3: float):
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
    Vp0 = np.sqrt(c33 / density_gcm3)
    Vs0 = np.sqrt(c55 / density_gcm3)

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


def HaoStovas_params(cij: np.ndarray, density_gcm3: float):
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
    Vp0 = np.sqrt(c33 / density_gcm3)

    # estimate Hao-Stovas-Alkhalifah dimensionless parameters
    epsilon1 = np.sqrt((c22*(c33 - c44)) / (c23**2 + 2*c23*c44 + c33*c44))
    epsilon2 = np.sqrt((c11*(c33 - c55)) / (c13**2 + 2*c13*c55 + c33*c55))
    epsilon3 = np.sqrt((c22*(c11 - c66)) / (c12**2 + 2*c12*c66 + c11*c66))
    r1 = (c23**2 + 2*c23*c44 + c33*c44) / (c33*(c33 - c44))
    r2 = (c13**2 + 2*c13*c55 + c33*c55) / (c33*(c33 - c55))

    return Vp0, epsilon1, epsilon2, epsilon3, r1, r2


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
