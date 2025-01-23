# -*- coding: utf-8 -*-
###############################################################################
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: averaging_schemes.py                                              #
# Description: This module calculates averaging elastic properties of solids. #
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

    # Validate the shape of elastic_tensor
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


def voigt_CPO_weighted_average(elastic_tensor: np.ndarray,
                               ODF: np.ndarray):
    """Calculates the elastic tensor Voigt average of a
    mineral phase considering the crystallographic
    preferred orientation of the aggregate

    The Voigt average is defined as the weighted arithmetic
    mean of elastic tensors, that is:

    C_ij_voigt = Σ_n (ODF_n * C_nij)

    This can be calculated using Einstein summation (np.einsum)
    using the notation 'n, nij -> ij' where:
    'n' represents the array with the orientation weights
    'nij' represents the elastic tensors 3-d array, C_nij
    'ij' represents the output array, 2-d array

    Parameters
    ----------
    elastic_tensor : numpy array, shape(6, 6)
        elastic tensor of the mineral phase
    ODF : numpy array
        Orientation Distribution Function

    Returns
    -------
    Cij_voigt : numpy array, shape(6, 6)
        Voigt average elastic tensor for the aggregate
    """

    # Validate the shape of elastic_tensor
    assert elastic_tensor.shape[1:] == (6, 6), \
        f"array shape must be (n, 6, 6) not {elastic_tensor.shape}"

    pass


def reuss_CPO_weighted_average(compliance_tensor: np.ndarray,
                               ODF: np.ndarray):
    """Calculates the compliance tensor Reuss average of a
    mineral phase considering the crystallographic
    preferred orientation of the aggregate

    The Reuss average is defined as the weighted arithmetic
    mean of elastic tensors, that is:

    TODO

    Parameters
    ----------
    compliance_tensor : numpy array, shape(6, 6)
        compliance tensor of the mineral phase
    ODF : numpy array
        Orientation Distribution Function

    Returns
    -------
    Cij_reuss : numpy array, shape(6, 6)
        Reuss average elastic tensor for the aggregate
        material.
    """
    pass


def rotate_tensor(rot_matrix, tensor):
    """
    Rotate a 6x6 elasticity tensor using a rotation matrix.

    Parameters
    ----------
    rot_matrix :
        3x3 rotation matrix
    tensor :
        6x6 elasticity tensor

    Returns
    -------
    rotated_tensor :
        6x6 rotated elasticity tensor
    """
    # Create a transformation matrix for tensor components
    rot_tensor = np.kron(rot_matrix, rot_matrix)

    # Rotate the tensor using the transformation matrix
    rotated_tensor = np.dot(np.dot(rot_tensor, tensor), rot_tensor.T)

    return rotated_tensor


def weighted_average_rotated_tensors(rotation_matrices, elasticity_tensor, weights):
    """
    TODO
    Calculate the weighted average of rotated elasticity tensors.

    Parameters
    ----------
    rotation_matrices :
        List of 3x3 rotation matrices
    elasticity_tensor :
        6x6 elasticity tensor
    weights :
        array of weights for rotation matrices

    Returns
    -------
    weighted_average_tensor :
        6x6 weighted average rotated elasticity tensor
    """

    # Ensure the weights sum up to 1
    if not np.isclose(np.sum(weights), 1):
        weights = weights / np.sum(weights)

    # Stack the rotation matrices into a 3D array
    rotation_matrices = np.array(rotation_matrices)  # shape: (num_matrices, 3, 3)

    # Expand dimensions to enable broadcasting
    expanded_rotation_matrices = rotation_matrices[:, np.newaxis, :, :]  # shape: (num_matrices, 1, 3, 3)

    # Expand dimensions of the elasticity tensor for broadcasting
    expanded_elasticity_tensor = elasticity_tensor[np.newaxis, :, :]  # shape: (1, 6, 6)

    # Perform element-wise tensor multiplication
    rotated_tensors = np.einsum('aijb,aik->akjb', expanded_rotation_matrices, expanded_elasticity_tensor)

    # Calculate the weighted sum of the rotated tensors
    weighted_average_tensor = np.sum(weights[:, np.newaxis, np.newaxis, np.newaxis] * rotated_tensors, axis=0)

    return weighted_average_tensor

# End of file
