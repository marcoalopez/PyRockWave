# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: coordinates.py                                                    #
# Description: Functions for converting between coordinate systems and        #
# generating equispaced spherical grids.                                      #
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
import numpy.typing as npt


# Function definitions
def sph2cart(
    azimuth_rad: npt.ArrayLike,
    polar_rad: npt.ArrayLike,
    r: npt.ArrayLike = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert from spherical/polar (magnitude, azimuth, polar) to
    cartesian coordinates. Azimuth and polar angles are as used in
    physics (ISO 80000-2:2019) and in radians. If the polar angle is
    not given, the coordinate is assumed to lie on the XY plane.

    Parameters
    ----------
    azimuth_rad : int, float or array_like
        Azimuth angle with respect to the x-axis direction in radians,
        values between 0 and 2*pi.
    polar_rad : int, float or array_like
        Polar angle with respect to the zenith (z) direction in radians,
        values between 0 and pi.
    r : int, float or array_like, optional
        Radial distance (magnitude of the vector), defaults to 1.

    Returns
    -------
    x : np.ndarray
        Cartesian x coordinates.
    y : np.ndarray
        Cartesian y coordinates.
    z : np.ndarray
        Cartesian z coordinates.
    """

    x = r * np.sin(polar_rad) * np.cos(azimuth_rad)
    y = r * np.sin(polar_rad) * np.sin(azimuth_rad)
    z = r * np.cos(polar_rad)

    return x, y, z


def cart2sph(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D rectangular cartesian coordinates to spherical
    coordinates.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Returns
    -------
    r : np.ndarray
        Radial distance.
    theta : np.ndarray
        Inclination/polar angle in radians, range [0, pi].
    phi : np.ndarray
        Azimuthal angle in radians, range [0, 2*pi].

    Notes
    -----
    This function follows the ISO 80000-2:2019 norm (physics convention).
    The input coordinates (x, y, z) are assumed to be in a right-handed
    Cartesian system. The spherical coordinates are returned in the order
    (r, theta, phi). The angles theta and phi are in radians.

    When the input is the origin (0, 0, 0), r is 0 and theta returns NaN.
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi)

    return r, theta, phi


def equispaced_S2_grid(
    num_points: int = 20809,
    degrees: bool = False,
    hemisphere: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns an approximately equispaced spherical grid in
    spherical coordinates (azimuthal and polar angles) using
    a modified version of the offset Fibonacci lattice algorithm.
    The Fibonacci Lattice algorithm is often considered one of the
    best in terms of balancing uniformity, computational efficiency,
    and ease of implementation. It's particularly good for large
    numbers of points.

    Note: Mathematically speaking, you cannot put more than 20
    perfectly evenly spaced points on a sphere. However, there
    are good-enough ways to approximately position evenly
    spaced points on a sphere.

    See also:
    https://arxiv.org/pdf/1607.04590.pdf
    https://github.com/gradywright/spherepts

    Parameters
    ----------
    num_points : int, optional
        The number of points, by default 20809.
    degrees : bool, optional
        Whether you want angles in degrees or radians,
        by default False (=radians).
    hemisphere : None, 'upper' or 'lower', optional
        Whether you want the grid to be distributed over the entire
        sphere, the upper hemisphere, or the lower hemisphere.
        Defaults to None (full sphere).

    Returns
    -------
    azimuth : np.ndarray
        Azimuthal angles sorted by polar angle. In radians if
        degrees=False, in degrees otherwise.
    polar_ang : np.ndarray
        Polar angles sorted in ascending order. In radians if
        degrees=False, in degrees otherwise.

    Raises
    ------
    ValueError
        If num_points is not a positive integer, or if hemisphere
        is not one of None, 'upper', or 'lower'.
    """

    if not isinstance(num_points, int) or num_points < 1:
        raise ValueError(f"num_points must be a positive integer, got {num_points!r}")
    if hemisphere not in (None, "upper", "lower"):
        raise ValueError(
            f"hemisphere must be None, 'upper', or 'lower', got {hemisphere!r}"
        )

    # Subtract 2 to account for the two pole points added manually below.
    # When filtering to a hemisphere, double first so the filtered half
    # still contains the requested number of points.
    if hemisphere is None:
        n = num_points - 2
    else:
        n = (num_points * 2) - 2

    epsilon = _set_epsilon(n)

    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(0, n)

    # compute azimuthal and polar angles in radians
    azimuth = 2 * np.pi * i / golden_ratio
    polar_ang = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))

    # place one datapoint at each pole
    azimuth = np.insert(azimuth, 0, 0)
    polar_ang = np.insert(polar_ang, 0, 0)
    azimuth = np.append(azimuth, 0)
    polar_ang = np.append(polar_ang, np.pi)

    # apply hemisphere filter
    if hemisphere == "upper":
        mask = polar_ang <= np.pi / 2
    elif hemisphere == "lower":
        mask = polar_ang >= np.pi / 2
    else:
        mask = np.ones(len(polar_ang), dtype=bool)

    azimuth = azimuth[mask] % (2 * np.pi)
    polar_ang = polar_ang[mask]

    if degrees:
        azimuth = np.rad2deg(azimuth)
        polar_ang = np.rad2deg(polar_ang)

    # order angles according to polar angle, from 0 to 180 (degrees)
    azimuth = azimuth[np.argsort(polar_ang)]
    polar_ang = np.sort(polar_ang)

    return azimuth, polar_ang


def equispaced_S2_grid_fsa(num_points: int) -> np.ndarray:
    """
    Generate evenly distributed points on a unit sphere using
    the Fibonacci sphere algorithm (sunflower mapping).

    The algorithm distributes points using the golden angle as the
    azimuthal increment and equal z-spacing (via arccos) to approximate
    a uniform distribution from the south pole to the north pole.

    Parameters
    ----------
    num_points : int
        The number of points to generate on the sphere surface.

    Returns
    -------
    np.ndarray
        An array of shape (num_points, 3) containing the (x, y, z)
        coordinates of the generated points on the unit sphere (r=1).

    Raises
    ------
    ValueError
        If num_points is not a positive integer.

    Notes
    -----
    The golden angle phi = pi*(3 - sqrt(5)) ensures azimuthal increments
    that avoid radial clustering. All points lie on the unit sphere (r=1).

    References
    ----------
    Swinbank, R., & Purser, R.J. (2006). Fibonacci grids: A novel approach to
    global modelling. Quarterly Journal of the Royal Meteorological Society,
    132(619), 1769-1793.
    """

    if not isinstance(num_points, int) or num_points < 1:
        raise ValueError(
            f"num_points must be a positive integer, got {num_points!r}"
        )

    # Generate indices from 1 to num_points
    indices = np.arange(1, num_points + 1)

    # Calculate the golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Calculate the polar angle (latitude)
    polar_angle = np.arccos(1 - 2 * indices / (num_points + 1))

    # Calculate the azimuthal angle (longitude)
    azimuthal_angle = golden_angle * indices

    # Convert spherical coordinates to Cartesian coordinates
    x, y, z = sph2cart(azimuthal_angle, polar_angle)

    # Stack the coordinates into a single array
    sphere_points = np.column_stack((x, y, z))

    return sphere_points


# =================================================================
# Private helpers for internal use only

def _set_epsilon(n: int) -> float:
    """Internal function used by equispaced_S2_grid.
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
