# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: averaging_schemes.py                                              #
# Description: This module calculates averaging elastic properties of solids. #
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
# =========================================================================== #

# Import statements
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tensor_tools import _rearrange_tensor, _tensor_in_voigt


# Function definitions

def voigt_volume_weighted_average(
    elastic_tensors: np.ndarray,
    volume_fractions: np.ndarray
) -> np.ndarray:
    """
    Calculates the Voigt average of a set of mineral
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

    volume_fractions = _validate_volume_weighted_average(
        elastic_tensors, volume_fractions
    )

    return np.einsum('n,nij->ij', volume_fractions, elastic_tensors)


def reuss_volume_weighted_average(
    elastic_tensors: np.ndarray,
    volume_fractions: np.ndarray
) -> np.ndarray:
    """
    Calculates the Reuss average of a set of mineral
    phases, described by their elastic tensors and
    volume fractions.

    The Reuss average is defined as the weighted arithmetic
    mean of compliance tensors, that is:

    S_ij_reuss = Σ_n (F_n * S_nij)

    This can be calculated using Einstein summation (np.einsum)
    using the notation 'n, nij -> ij' where:
    'n' represents the array with the volume fractions, F_n
    'nij' represents the compliance tensors 3-d array, S_nij
    'ij' represents the output array, 2-d array

    The compliance tensor is the inverse of the elastic tensor.
    The Reuss average elastic tensor is then the inverse of the
    Reuss average compliance tensor.

    Parameters
    ----------
    elastic_tensors : numpy array, shape(n, 6, 6)
        elastic tensors of constitutive phases for
        which to calculate the Reuss average.
    volume_fractions : numpy array, shape(n,)
        Fraction of each constitutive phase in the
        composite material.

    Returns
    -------
    Cij_reuss : numpy array, shape(6, 6)
        Reuss average elastic tensor for the composite
        material.
    """

    volume_fractions = _validate_volume_weighted_average(
        elastic_tensors, volume_fractions
    )

    # calculate compliance tensors
    compliance_tensors = np.linalg.inv(elastic_tensors)

    # Calculate the Reuss average compliance tensor
    Sij_reuss = np.einsum('n,nij->ij', volume_fractions, compliance_tensors)

    return np.linalg.inv(Sij_reuss)


def voigt_CPO_weighted_average(
    elastic_tensor: np.ndarray,
    ODF: pd.DataFrame
) -> np.ndarray:
    """
    Calculates the Voigt average elastic tensor of a mineral
    phase considering the crystallographic preferred orientation
    (CPO) of the aggregate.

    The Voigt average is the weighted arithmetic mean of the
    single-crystal elastic tensor rotated to each discrete
    orientation present in the ODF:

    C_ij_voigt = Σ_n (ODF_n * C_nij)

    where C_nij is the elastic tensor rotated to orientation n
    and ODF_n is the corresponding volume fraction. Rotations
    are applied in the full 4th-rank tensor space via

    C'_ijkl = R_ia R_jb R_kc R_ld C_abcd

    using the Bunge zxz (extrinsic) Euler angle convention, then
    mapped back to Voigt notation.

    Parameters
    ----------
    elastic_tensor : numpy array, shape(6, 6)
        Single-crystal elastic tensor of the mineral phase in
        Voigt notation.
    ODF : pd.DataFrame
        Orientation Distribution Function. Must have at least
        four columns: the first three are the Euler angles
        (phi1, Phi, phi2) in degrees using the Bunge zxz
        extrinsic convention; the fourth is the volume fraction
        (or percentage) associated with each orientation.

    Returns
    -------
    Cij_voigt : numpy array, shape(6, 6)
        Voigt average elastic tensor for the aggregate.
    """

    weights = _validate_CPO_weighted_average(elastic_tensor, ODF)

    euler_angles = ODF.iloc[:, :3].to_numpy(dtype=float)
    rot_matrices = R.from_euler('zxz', euler_angles, degrees=True).as_matrix()

    C_ijkl = _rearrange_tensor(elastic_tensor)

    # Rotate single-crystal tensor to all N orientations at once:
    # C'_nijkl = R_nia R_njb R_nkc R_nld C_abcd
    C_rotated = np.einsum(
        'nia,njb,nkc,nld,abcd->nijkl',
        rot_matrices, rot_matrices, rot_matrices, rot_matrices,
        C_ijkl
    )

    C_weighted = np.einsum('n,nijkl->ijkl', weights, C_rotated)

    return _tensor_in_voigt(C_weighted)


def reuss_CPO_weighted_average(
    elastic_tensor: np.ndarray,
    ODF: pd.DataFrame
) -> np.ndarray:
    """
    Calculates the Reuss average elastic tensor of a mineral
    phase considering the crystallographic preferred orientation
    (CPO) of the aggregate.

    The Reuss average is the weighted arithmetic mean of the
    single-crystal compliance tensor rotated to each discrete
    orientation present in the ODF:

    S_ij_reuss = Σ_n (ODF_n * S_nij)

    where S_nij is the compliance tensor rotated to orientation n
    and ODF_n is the corresponding volume fraction. The Reuss
    average elastic tensor is then the inverse of S_ij_reuss.
    Because rotation commutes with matrix inversion for orthogonal
    matrices, the compliance tensor is rotated directly:

    S'_ijkl = R_ia R_jb R_kc R_ld S_abcd

    using the Bunge zxz (extrinsic) Euler angle convention, then
    mapped back to Voigt notation before inversion.

    Parameters
    ----------
    elastic_tensor : numpy array, shape(6, 6)
        Single-crystal elastic tensor of the mineral phase in
        Voigt notation.
    ODF : pd.DataFrame
        Orientation Distribution Function. Must have at least
        four columns: the first three are the Euler angles
        (phi1, Phi, phi2) in degrees using the Bunge zxz
        extrinsic convention; the fourth is the volume fraction
        (or percentage) associated with each orientation.

    Returns
    -------
    Cij_reuss : numpy array, shape(6, 6)
        Reuss average elastic tensor for the aggregate.
    """

    weights = _validate_CPO_weighted_average(elastic_tensor, ODF)

    euler_angles = ODF.iloc[:, :3].to_numpy(dtype=float)
    rot_matrices = R.from_euler('zxz', euler_angles, degrees=True).as_matrix()

    S_ijkl = _rearrange_tensor(np.linalg.inv(elastic_tensor))

    # Rotate single-crystal compliance tensor to all N orientations at once:
    # S'_nijkl = R_nia R_njb R_nkc R_nld S_abcd
    S_rotated = np.einsum(
        'nia,njb,nkc,nld,abcd->nijkl',
        rot_matrices, rot_matrices, rot_matrices, rot_matrices,
        S_ijkl
    )

    S_weighted = np.einsum('n,nijkl->ijkl', weights, S_rotated)

    return np.linalg.inv(_tensor_in_voigt(S_weighted))


# =================================================================
# Private helpers for internal use only

def _validate_volume_weighted_average(
    elastic_tensors: np.ndarray,
    volume_fractions: np.ndarray
) -> np.ndarray:
    """
    Validate and normalise inputs shared by both volume-weighted
    averaging functions. Returns a normalised copy of volume_fractions.

    Raises
    ------
    ValueError
        If shapes are incompatible, any fraction is negative, or
        elastic_tensors is not a 3-D array of (6, 6) matrices.
    """
    if elastic_tensors.ndim != 3 or elastic_tensors.shape[1:] != (6, 6):
        raise ValueError(
            f"elastic_tensors must have shape (n, 6, 6), got {elastic_tensors.shape}"
        )
    if volume_fractions.ndim != 1:
        raise ValueError(
            f"volume_fractions must be 1-D, got shape {volume_fractions.shape}"
        )
    if elastic_tensors.shape[0] != volume_fractions.shape[0]:
        raise ValueError(
            f"Number of tensors ({elastic_tensors.shape[0]}) must match "
            f"number of volume fractions ({volume_fractions.shape[0]})"
        )
    if np.any(volume_fractions < 0):
        raise ValueError("All volume fractions must be non-negative")
    if not np.isclose(volume_fractions.sum(), 1.0):
        print("Volume fractions do not add up to 1, recalculating...")
        volume_fractions = volume_fractions / volume_fractions.sum()
    return volume_fractions


def _validate_CPO_weighted_average(
    elastic_tensor: np.ndarray,
    ODF: pd.DataFrame
) -> np.ndarray:
    """
    Validate inputs shared by both CPO-weighted averaging functions.
    Returns a normalised 1-D array of ODF volume fractions.

    Raises
    ------
    ValueError
        If elastic_tensor is not a (6, 6) array, ODF is not a DataFrame
        with at least 4 columns, or any weight is negative.
    """
    if not isinstance(elastic_tensor, np.ndarray) or elastic_tensor.shape != (6, 6):
        raise ValueError(
            f"elastic_tensor must have shape (6, 6), got {elastic_tensor.shape}"
        )
    if not isinstance(ODF, pd.DataFrame):
        raise TypeError("ODF must be a pandas DataFrame")
    if ODF.shape[1] < 4:
        raise ValueError(
            f"ODF DataFrame must have at least 4 columns, got {ODF.shape[1]}"
        )
    weights = ODF.iloc[:, 3].to_numpy(dtype=float)
    if np.any(weights < 0):
        raise ValueError("All ODF volume fractions must be non-negative")
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    return weights


# End of file
