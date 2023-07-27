# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: averaging_schemes.py                                      #
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
def voigt_average(Cijs, volume_fractions):
    """
    Calculates the Voigt average of a set of material
    phases, described by their elastic tensors and
    volume fractions.

    The Voigt average is defined as the weighted arithmetic
    mean of elastic tensors, that is:

    Cij_voigt = f1 * Cij_1 + f2 * Cij_2 + ... + fn * Cij_n

    Parameters
    ----------
    Cijs : numpy array, shape(n, 6, 6)
        Stiffness tensors of constitutive phases for
        which to calculate the Voigt average.
    volume_fractions : numpy array, shape(n,)
        Fraction of each constitutive phase in the
        composite material.

    Returns
    -------
    Cij_voigt : numpy array, shape(6, 6)
        Voigt average elastic tensor for the composite
        material.
    """

    # Validate volume fraction sums to unity
    if not np.isclose(np.sum(volume_fractions), 1):
        volume_fractions = volume_fractions / np.sum(volume_fractions)

    # Calculate the Voigt average using einsum
    voigt_average_tensor = np.einsum('i, ijk -> jk', volume_fractions, Cijs)

    return voigt_average_tensor

# End of file
