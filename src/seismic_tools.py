# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/

import numpy as np
import pandas as pd


def Thomsen_params(cij: np.ndarray, density: float):
    """Estimate the Thomsen paramenters for
    weak polar anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material.
    density : float
        The density of the material.

    Returns
    -------
    [float, float, float, float, float]
        Tuple/List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(cij)
    c13 = cij[0, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c44 / density)

    # estimate Thomsen dimensionless parameters
    epsilon = (c11 - c33) / (2 * c33)
    delta = ((c13 + c44)**2 - (c33 - c44)**2) / (2 * c33 * (c33 - c44))
    gamma = (c66 - c44) / (2 * c44)

    return Vp0, Vs0, epsilon, delta, gamma


def Tsvankin_params(cij: np.ndarray, density: float):
    """Estimate the Tsvankin paramenters for
    azimuthal orthotropic anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material.
    density : float
        The density of the material.

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


def equispaced_S2_grid(n=20809, degrees=False, hemisphere=None):
    """Returns an approximately equispaced spherical grid in
    spherical coordinates (azimuthal and polar angles) using
    a modified version of the offset Fibonacci lattice algorithm

    Note: Matemathically speaking, you cannot put more than 20
    (exactly) evenly spaced—some points on a sphere. However,
    there are good-enough ways to approximately position evenly
    spaced points on a sphere.

    See also:
    https://arxiv.org/pdf/1607.04590.pdf
    https://github.com/gradywright/spherepts

    Parameters
    ----------
    n : int, optional
        the number of points, by default 20809

    degrees : bool, optional
        whether you want angles in degrees or radians,
        by default False (=radians)

    hemisphere : None, 'upper' or 'lower'
        whether you want the grid to be distributed
        over the entire sphere, over the upper
        hemisphere, or over the lower hemisphere.

    Returns
    -------
    _type_
        _description_
    """

    # set sample size
    if hemisphere is None:
        n = n - 2
    else:
        n = (n * 2) - 2

    # get epsilon value based on sample size
    epsilon = _set_epsilon(n)

    golden_ratio = (1 + 5**0.5)/2
    i = np.arange(0, n)

    # estimate polar (phi) and theta (azimutal) angles
    theta = 2 * np.pi * i / golden_ratio  # in degrees
    phi = np.arccos(1 - 2*(i + epsilon) / (n - 1 + 2*epsilon))  # in rad

    # place a datapoint at each pole, it adds two datapoints removed before
    theta = np.insert(theta, 0, 0)
    theta = np.append(theta, 0)
    phi = np.insert(phi, 0, 0)
    phi = np.append(phi, np.pi)

    if degrees is False:
        if hemisphere == 'upper':
            return np.deg2rad(theta[phi <= np.pi/2] % 360), phi[phi <= np.pi/2]
        elif hemisphere == 'lower':
            return np.deg2rad(theta[phi >= np.pi/2] % 360), phi[phi >= np.pi/2]
        else:
            return np.deg2rad(theta % 360), phi
    else:
        if hemisphere == 'upper':
            return theta[phi <= np.pi/2] % 360, np.rad2deg(phi[phi <= np.pi/2])
        elif hemisphere == 'lower':
            return theta[phi >= np.pi/2] % 360, np.rad2deg(phi[phi >= np.pi/2])
        else:
            return theta % 360, np.rad2deg(phi)


def weak_polar_anisotropy(elastic):
    """Estimate the speed of body waves in a material as a function
    of propagation direction assuming that the material have a
    weak polar anisotropy using the Thomsen approach (Thomsen, 1986).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using the weak polar
        anisotropy model.
    """

    # generate equispaced spherical coordinates
    azimuths, polar = equispaced_S2_grid(n=80_000, hemisphere='upper')

    # get Thomsen parameters
    Vp0, Vs0, epsilon, delta, gamma = Thomsen_params(elastic.Cij, elastic.density)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = Vp0 * (1 + delta * sin_theta**2 * cos_theta**2 + epsilon * sin_theta**4)
    Vsv = Vs0 * (1 + (Vp0 / Vs0)**2 * (epsilon - delta) * sin_theta**2 * cos_theta**2)
    Vsh = Vs0 * (1 + gamma * sin_theta**2)

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp,
            'Vsv': Vsv,
            'Vsh': Vsh}

    return pd.DataFrame(data)


def polar_anisotropy(elastic):
    """Estimate the speed of body waves in a material as a function
    of propagation direction assuming that the material have a
    polar anisotropy using the Anderson approach (Anderson, 1961).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using a polar
        anisotropy model.
    """

    # generate equispaced spherical coordinates
    azimuths, polar = equispaced_S2_grid(n=80_000, hemisphere='upper')

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(elastic.Cij)
    c13 = elastic.Cij[0, 2]

    # estimate D value
    first_term = (c33 - c44)**2
    second_term = 2 * (2 * (c13 + c44)**2 - (c33 - c44)*(c11 + c33 - 2*c44)) * np.sin(polar)**2
    third_term = ((c11+c33-2*c44)**2 - 4*(c13+c44)**2) * np.sin(polar)**4
    D = np.sqrt(first_term + second_term + third_term)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 + D))
    Vsv = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 - D))
    Vsh = np.sqrt((1 / elastic.density) * (c44 * cos_theta**2 + c66 * sin_theta**2))

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp,
            'Vsv': Vsv,
            'Vsh': Vsh}

    return pd.DataFrame(data)


def orthotropic_azimuthal_anisotropy(elastic):
    """The simplest realistic case of azimuthal anisotropy is that of
    orthorhombic anisotropy (a.k.a. orthotropic).

    Parameters
    ----------
    elastic : _type_
        _description_
    """
    pass


####################################################################
# The following functions are for internal use by the script only
# and are not intended to be used directly by the user.
####################################################################

def _set_epsilon(n):
    """Internal method used by the funtion
    equispaced_S2_grid.
    """
    if n >= 600_000:
        return 214
    elif n >= 400_000:
        return 75
    elif n >= 11_000:
        return 27
    elif n >= 890:
        return 10
    elif n >= 177:
        return 3.33
    elif n >= 24:
        return 1.33
    else:
        return 0.33


def _symmetrise_tensor(tensor):
    return 0.5 * (tensor + tensor.T)


def _rearrange_tensor(C_ij):
    """Turn a 6x6 elastic tensor into a 3x3x3x3 tensor
    according to Voigt notation.
    """
    voigt_notation = {0: (0, 0), 11: (1, 1), 22: (2, 2),
                      12: (1, 2), 21: (2, 1), 2: (2, 0),
                      20: (2, 0), 1: (1, 0), 10: (1, 0)}

    C_ijkl = np.zeros((3, 3, 3, 3))

    for key, value in voigt_notation.items():
        i, j = value[0], value[1]
        for key, value in voigt_notation.items():
            k, l = value[0], value[1]
            C_ijkl[i, j, k, l] = C_ij[key]

    return C_ijkl


def _hessian_christoffelmat(C):
    """
    Computes the Hessian of the Christoffel matrix.
    hessianmat[i][j][k][l] = d^2 M_kl / dx_i dx_j (note the indices).
    """
    hessianmat = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for L in range(3):
                    hessianmat[i][j][k][L] = C[k][i][j][L] + C[k][j][i][L]

    return hessianmat
