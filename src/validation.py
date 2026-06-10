# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: validation.py                                                     #
# Description: Shared input validation helpers for elastic tensors            #
# and wavevectors.                                                            #
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
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
# =========================================================================== #

# Import statements
import numpy as np


# Function definitions
def validate_cij(Cij: np.ndarray) -> bool:
    """
    Validate a 6x6 Voigt stiffness matrix.

    The matrix must be a symmetric 6x6 NumPy array and positive
    definite (Born mechanical stability criterion). Positive
    definiteness guarantees that all Christoffel eigenvalues, and
    hence all phase velocities, are real and positive.

    Parameters
    ----------
    Cij : numpy.ndarray
        The elastic stiffness tensor in Voigt notation.

    Returns
    -------
    bool
        True if Cij passes all checks.

    Raises
    ------
    ValueError
        If Cij is not a 6x6 NumPy array, is not symmetric, or is
        not positive definite.
    """
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij should be a 6x6 NumPy array.")
    if not np.allclose(Cij, Cij.T):
        raise ValueError("Cij should be symmetric.")
    if np.linalg.eigvalsh(Cij).min() <= 0:
        raise ValueError(
            "Cij is not positive definite, i.e. the elastic tensor "
            "is mechanically unstable (Born stability criterion)."
        )
    return True


def validate_wavevectors(wavevectors: np.ndarray) -> bool:
    """
    Validate a wavevector array.

    Parameters
    ----------
    wavevectors : numpy.ndarray of shape (3,) or (n, 3)
        The wavevector array to validate.

    Returns
    -------
    bool
        True if the array has an acceptable shape and type.

    Raises
    ------
    ValueError
        If the array is not a NumPy array or does not have shape
        (3,) or (n, 3).
    """
    if not isinstance(wavevectors, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if wavevectors.ndim not in [1, 2]:
        raise ValueError("Input array must be 1-dimensional or 2-dimensional.")

    if wavevectors.ndim == 1 and wavevectors.shape != (3,):
        raise ValueError("1-dimensional array must have shape (3,).")

    if wavevectors.ndim == 2 and wavevectors.shape[1] != 3:
        raise ValueError("2-dimensional array must have shape (n, 3).")

    return True

# End of file
