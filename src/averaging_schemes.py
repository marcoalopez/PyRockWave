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
def voigt_volume_weighted_average(elastic_tensors: np.ndarray,
                                  volume_fractions: np.ndarray):
    """Calculates the Voigt average of a set of mineral
    phases, described by their elastic tensors and
    volume fractions.

    The Voigt average is defined as the weighted arithmetic
    mean of elastic tensors, that is:

    C_ij_voigt = Σ_n (F_n * C_nij)

    This can be calculated using Einstein summation (np.einsum)
    using the notation 'n, nij -> ij' where:
    'n' represents the array with the volume fractions, F_n
    'nij' represents the elastic tensors 3-d array, C_nij
    'ij' represents the output array, 2-d array

    Parameters
    ----------
    elastic_tensors : numpy array, shape(n, 6, 6)
        elastic tensors of constitutive phases for
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

    # Validate the shape of elastic_tensors
    assert elastic_tensors.shape[1:] == (6, 6), \
        f"array shape must be (n, 6, 6) not {elastic_tensors.shape}"

    # Validate that volume fractions sums to unity
    if not np.isclose(np.sum(volume_fractions), 1):
        volume_fractions = volume_fractions / np.sum(volume_fractions)

    # Calculate the Voigt average
    Cij_voigt = np.einsum('n, nij -> ij', volume_fractions, elastic_tensors)

    return Cij_voigt


def reuss_volume_weighted_average(elastic_tensors: np.ndarray,
                                  volume_fractions: np.ndarray):
    """Calculates the Reuss average of a set of mineral
    phases, described by their elastic tensors and
    volume fractions.

    The Ruess average is defined as the weighted arithmetic
    mean of compliance tensors, that is:

    S_ij_reuss = Σ_n (F_n * S_nij)

    This can be calculated using Einstein summation (np.einsum)
    using the notation 'n, nij -> ij' where:
    'n' represents the array with the volume fractions, F_n
    'nij' represents the compliance tensors 3-d array, S_nij
    'ij' represents the output array, 2-d array

    The compliance tensor is the inverse of the elastic tensor

    Parameters
    ----------
    elastic_tensors : numpy array, shape(n, 6, 6)
        elastic tensors of constitutive phases for
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

    # Validate the shape of elastic_tensors
    assert elastic_tensors.shape[1:] == (6, 6), \
        f"array shape must be (n, 6, 6) not {elastic_tensors.shape}"

    # Validate that volume fractions sums to unity
    if not np.isclose(np.sum(volume_fractions), 1):
        volume_fractions = volume_fractions / np.sum(volume_fractions)

    # calculate compliance tensors
    compliance_tensors = np.linalg.inv(elastic_tensors)

    # Calculate the Voigt average
    Sij_reuss = np.einsum('n, nij -> ij', volume_fractions, compliance_tensors)

    return Sij_reuss

# End of file
