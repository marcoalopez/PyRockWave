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
    num_points: int = 41_253,
    hemisphere: str | None = None,
    include_axes: bool = False,
) -> np.ndarray:
    """
    Generate an approximately equispaced grid of unit vectors on the
    sphere using the Fibonacci sphere algorithm (sunflower mapping).

    The algorithm distributes points using the golden angle as the
    azimuthal increment and equal z-spacing (via arccos) to approximate
    a uniform distribution from pole to pole. The mean angular spacing
    between neighbouring points is approximately sqrt(4*pi/num_points)
    radians; the default of 41,253 points gives ~1 degree.

    This is the recommended function for generating wavevector grids
    for the christoffel module: it returns Cartesian unit vectors of
    shape (n, 3), which the christoffel functions consume directly,
    and it has no under-sampled ring around the poles (compare
    equispaced_S2_grid_offset).

    Parameters
    ----------
    num_points : int, optional
        The number of lattice points to generate, by default 41253
        (~1 degree mean spacing). When a hemisphere is requested, the
        returned grid contains num_points points on that hemisphere.
    hemisphere : None, 'upper' or 'lower', optional
        Whether to distribute the points over the full sphere (None,
        the default), the upper hemisphere (z > 0), or the lower
        hemisphere (z < 0).
    include_axes : bool, optional
        If True, prepend the coordinate-axis directions ±z, ±x, ±y
        (those compatible with the requested hemisphere) to the grid.
        The lattice never samples these directions exactly, yet they
        often coincide with symmetry axes where seismic velocity
        extrema occur. Defaults to False.

    Returns
    -------
    np.ndarray
        An array of shape (m, 3) containing the (x, y, z) coordinates
        of the points on the unit sphere (r=1), where m equals
        num_points plus, if include_axes=True, the number of prepended
        axis directions (6 for the full sphere, 5 for a hemisphere).

    Raises
    ------
    ValueError
        If num_points is not a positive integer, or if hemisphere
        is not one of None, 'upper', or 'lower'.

    Notes
    -----
    The golden angle phi = pi*(3 - sqrt(5)) ensures azimuthal
    increments that avoid radial clustering. Hemisphere selection
    generates a full-sphere lattice of 2*num_points points and keeps
    the exact half lying on the requested side (no lattice point falls
    on the equator for an even number of points).

    References
    ----------
    Swinbank, R., & Purser, R.J. (2006). Fibonacci grids: A novel approach to
    global modelling. Quarterly Journal of the Royal Meteorological Society,
    132(619), 1769-1793. https://doi.org/10.1256/qj.05.227
    """

    if not isinstance(num_points, int) or num_points < 1:
        raise ValueError(
            f"num_points must be a positive integer, got {num_points!r}"
        )
    if hemisphere not in (None, "upper", "lower"):
        raise ValueError(
            f"hemisphere must be None, 'upper', or 'lower', got {hemisphere!r}"
        )

    # Generate a full-sphere lattice; double it when a hemisphere is
    # requested so the kept half contains num_points points.
    n = num_points if hemisphere is None else 2 * num_points
    indices = np.arange(1, n + 1)

    # Calculate the golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Calculate the polar (latitude) and azimuthal (longitude) angles
    polar_angle = np.arccos(1 - 2 * indices / (n + 1))
    azimuthal_angle = golden_angle * indices

    # Convert spherical coordinates to Cartesian coordinates
    x, y, z = sph2cart(azimuthal_angle, polar_angle)
    sphere_points = np.column_stack((x, y, z))

    # apply hemisphere filter (z is never exactly 0, see Notes)
    if hemisphere == "upper":
        sphere_points = sphere_points[sphere_points[:, 2] > 0]
    elif hemisphere == "lower":
        sphere_points = sphere_points[sphere_points[:, 2] < 0]

    if include_axes:
        axes = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        if hemisphere == "upper":
            axes = axes[axes[:, 2] >= 0]
        elif hemisphere == "lower":
            axes = axes[axes[:, 2] <= 0]
        sphere_points = np.vstack((axes, sphere_points))

    return sphere_points


def equispaced_S2_grid_offset(
    num_points: int = 20809,
    degrees: bool = False,
    hemisphere: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns an approximately equispaced spherical grid in
    spherical coordinates (azimuthal and polar angles) using
    a modified version of the offset Fibonacci lattice algorithm.
    The offsets are tuned to maximise the minimum distance between
    points (packing).

    Note: the packing-oriented offsets leave an under-sampled ring
    around each pole that grows relative to the mean point spacing
    for large num_points. For sampling directions (e.g. wavevector
    grids for the christoffel module), prefer equispaced_S2_grid,
    which is free of this artefact and returns Cartesian unit
    vectors directly. This function remains useful when spherical
    angles (optionally in degrees) are needed.

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


# =================================================================
# Private helpers for internal use only

def _set_epsilon(n: int) -> float:
    """
    Internal function used by equispaced_S2_grid_offset.
    """
    if n >= 40_000:
        return 25
    elif n >= 1000:
        return 10
    elif n >= 80:
        return 3.33
    else:
        return 2.66

def _calc_sample_size(ang_spacing_deg: float) -> int:
    """
    Estimate the number of approximately uniformly distributed points
    on the unit sphere for a given mean angular spacing.

    The estimate assumes

        θ ≈ sqrt(4π / N),

    where θ is the mean angular spacing (radians) and N is the number
    of points. Rearranging gives

        N ≈ 4π / θ².

    Note: This expression assumes each point represents an equal spherical
    area of approximately 4π / N steradians and that this area can be
    approximated by θ². This is accurate for small angular spacings
    (typically a few degrees or less) but becomes increasingly approximate
    for coarse samplings

    Parameters
    ----------
    ang_spacing_deg : float
        Desired mean angular spacing between neighbouring points, in
        degrees. Must be positive.

    Returns
    -------
    int
        Estimated number of sample points.

    Examples
    --------
    A spacing of 1° corresponds to approximately 41,253 points.
    """

    if ang_spacing_deg <= 0:
        raise ValueError("ang_spacing_deg must be positive.")

    ang_spacing_rad = np.deg2rad(ang_spacing_deg)
    sample_size = round(4 * np.pi / ang_spacing_rad**2)

    return sample_size

# End of file
