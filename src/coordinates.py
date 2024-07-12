# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: coordinates.py                                            #
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
def sph2cart(phi, theta, r=1):
    """ Convert from spherical/polar (magnitude, thetha, phi) to
    cartesian coordinates. Phi and theta angles are defined as in
    physics (ISO 80000-2:2019) and in radians.

    Parameters
    ----------
    phi : int, float or array with values between 0 and 2*pi
        azimuth angle respect to the x-axis direction in radians
    theta : int, float or array with values between 0 and pi/2,
        polar angle respect to the zenith (z) direction in radians
        optional
    r : int, float or array, optional
        radial distance (magnitud of the vector), defaults to 1

    Returns
    -------
    numpy ndarray (1d)
        three numpy 1d arrays with the cartesian x, y, and z coordinates
    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def cart2sph(x, y, z):
    """Converts from 3D cartesian to spherical coordinates.

    Parameters
    ----------
    x : int, float or array
        The x-coordinate(s) in Cartesian space.
    y : int, float or array
        The y-coordinate(s) in Cartesian space.
    z : int, float or array
        The z-coordinate(s) in Cartesian space.

    Returns
    -------
    numpy ndarray (1d)
        An array containing the polar coordinates (r, theta, phi)
        of the input Cartesian point, where r is the distance from
        the origin to the point, theta is the polar angle from the
        positive z-axis, and phi is the azimuthal angle from the
        positive x-axis (ISO 80000-2:2019).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, phi, theta


def equispaced_S2_grid(n=20809, degrees=False, hemisphere=None):
    """Returns an approximately equispaced spherical grid in
    spherical coordinates (azimuthal and polar angles) using
    a modified version of the offset Fibonacci lattice algorithm.
    The Fibonacci Lattice algorithm is often considered one of the
    best in terms of balancing uniformity, computational efficiency,
    and ease of implementation. It's particularly good for large
    numbers of points.

    Note: Matemathically speaking, you cannot put more than 20
    perfectly evenly spaced points on a sphere. However, there
    are good-enough ways to approximately position evenly
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

    golden_ratio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, n)

    # estimate polar (phi) and theta (azimutal) angles in radians
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))

    # place a datapoint at each pole, it adds two datapoints removed before
    theta = np.insert(theta, 0, 0)
    theta = np.append(theta, 0)
    phi = np.insert(phi, 0, 0)
    phi = np.append(phi, np.pi)

    if degrees is False:
        if hemisphere == 'upper':
            return theta[phi <= np.pi/2] % (2*np.pi), phi[phi <= np.pi/2]
        elif hemisphere == 'lower':
            return theta[phi >= np.pi/2] % (2*np.pi), phi[phi >= np.pi/2]
        else:
            return theta % (2*np.pi), phi
    else:
        if hemisphere == 'upper':
            return np.rad2deg(theta[phi <= np.pi/2]) % 360, np.rad2deg(phi[phi <= np.pi/2])
        elif hemisphere == 'lower':
            return np.rad2deg(theta[phi >= np.pi/2]) % 360, np.rad2deg(phi[phi >= np.pi/2])
        else:
            return np.rad2deg(theta) % 360, np.rad2deg(phi)


def equispaced_S2_grid2(num_points: int) -> np.ndarray:
    """
    Generate evenly distributed points on a unit sphere using
    the Equal-Area Partitioning method.

    This method, also known as the Spherical Fibonacci mapping,
    creates a spiral-like distribution of points from the south
    pole to the north pole of a unit sphere.

    Parameters
    ----------
    num_points : int
        The number of points to generate on the sphere surface.
    
    type : str
        Either in cartesian or spherical coordinates

    Returns
    -------
    np.ndarray
        An array of shape (num_points, 3) containing the (x, y, z)
        coordinates of the generated points on the unit sphere.

    Notes
    -----
    The algorithm uses the golden angle (phi) to determine the azimuthal angle
    increment between points, resulting in an approximately uniform distribution.

    References
    ----------
    Swinbank, R., & Purser, R.J. (2006). Fibonacci grids: A novel approach to
    global modelling. Quarterly Journal of the Royal Meteorological Society, 
    132(619), 1769-1793.
    """

    # Generate indices from 1 to num_points
    indices = np.arange(1, num_points + 1)

    # Calculate the golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Calculate the polar angle (latitude)
    polar_angle = np.arccos(1 - 2 * indices / (num_points + 1))

    # Calculate the azimuthal angle (longitude)
    azimuthal_angle = golden_angle * indices
    
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = np.cos(polar_angle)

    # Stack the coordinates into a single array
    sphere_points = np.column_stack((x, y, z))

    return sphere_points


####################################################################
# The following functions, starting with an underscore, are for
# internal use only, i.e. not intended to be used directly by
# the user.
####################################################################

def _set_epsilon(n):
    """Internal method used by the funtion
    equispaced_S2_grid.
    """
    if n >= 40_000:
        return 25
    elif n >= 1000:
        return 10
    elif n >= 80:
        return 3.33
    else:
        return 2.66

# End of file
