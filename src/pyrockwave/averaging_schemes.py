# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: averaging_schemes.py                                              #
# Description: This module calculates averaging elastic properties of solids. #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2026, Marco A. Lopez-Sanchez. All rights reserved.            #
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
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .utils.tensor_tools import _rearrange_tensor, _tensor_in_voigt


# Scaling factors relating the Voigt-notation compliance matrix to the
# 4th-rank compliance tensor: S_IJ = S_ijkl, 2*S_ijkl or 4*S_ijkl
# depending on whether none, one or both of I, J are in {4, 5, 6}.
# (No factors are needed for the stiffness tensor, where C_IJ = C_ijkl.)
_VOIGT_COMPLIANCE_FACTORS = np.ones((6, 6))
_VOIGT_COMPLIANCE_FACTORS[:3, 3:] = 2.0
_VOIGT_COMPLIANCE_FACTORS[3:, :3] = 2.0
_VOIGT_COMPLIANCE_FACTORS[3:, 3:] = 4.0


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

    where R is the crystal-to-sample rotation derived from the
    Bunge Euler angles (intrinsic ZXZ convention). The averaged
    tensor is mapped back to Voigt notation.

    Parameters
    ----------
    elastic_tensor : numpy array, shape(6, 6)
        Single-crystal elastic tensor of the mineral phase in
        Voigt notation.
    ODF : pd.DataFrame
        Orientation Distribution Function. Must have at least
        four columns: the first three are the Euler angles
        (phi1, Phi, phi2) in degrees following the Bunge
        convention (intrinsic ZXZ: rotations about the z, new x
        and new z axes); the fourth is the volume fraction
        (or percentage) associated with each orientation.

    Returns
    -------
    Cij_voigt : numpy array, shape(6, 6)
        Voigt average elastic tensor for the aggregate.
    """

    weights = _validate_CPO_weighted_average(elastic_tensor, ODF)

    euler_angles = ODF.iloc[:, :3].to_numpy(dtype=float)

    # Intrinsic ZXZ with (phi1, Phi, phi2) gives g^T, the Bunge
    # crystal-to-sample rotation
    rot_matrices = R.from_euler('ZXZ', euler_angles, degrees=True).as_matrix()

    C_ijkl = _rearrange_tensor(elastic_tensor)

    # Rotate single-crystal tensor to all N orientations at once:
    # C'_nijkl = R_nia R_njb R_nkc R_nld C_abcd
    C_rotated = np.einsum(
        'nia,njb,nkc,nld,abcd->nijkl',
        rot_matrices, rot_matrices, rot_matrices, rot_matrices,
        C_ijkl,
        optimize=True
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
    The compliance tensor of a rotated crystal equals the rotated
    compliance tensor, so the single-crystal compliance is rotated
    directly:

    S'_ijkl = R_ia R_jb R_kc R_ld S_abcd

    where R is the crystal-to-sample rotation derived from the
    Bunge Euler angles (intrinsic ZXZ convention). Because the
    Voigt-notation compliance matrix carries factors of 2 and 4
    on its shear-related entries relative to the tensor
    components, it is rescaled to true tensor components before
    rotation and back afterwards.

    Parameters
    ----------
    elastic_tensor : numpy array, shape(6, 6)
        Single-crystal elastic tensor of the mineral phase in
        Voigt notation.
    ODF : pd.DataFrame
        Orientation Distribution Function. Must have at least
        four columns: the first three are the Euler angles
        (phi1, Phi, phi2) in degrees following the Bunge
        convention (intrinsic ZXZ: rotations about the z, new x
        and new z axes); the fourth is the volume fraction
        (or percentage) associated with each orientation.

    Returns
    -------
    Cij_reuss : numpy array, shape(6, 6)
        Reuss average elastic tensor for the aggregate.
    """

    weights = _validate_CPO_weighted_average(elastic_tensor, ODF)

    euler_angles = ODF.iloc[:, :3].to_numpy(dtype=float)

    # Intrinsic ZXZ with (phi1, Phi, phi2) gives g^T, the Bunge
    # crystal-to-sample rotation
    rot_matrices = R.from_euler('ZXZ', euler_angles, degrees=True).as_matrix()

    # Remove the Voigt factors so that S_ijkl holds true tensor
    # components before applying the 4th-rank rotation law
    S_voigt = np.linalg.inv(elastic_tensor)
    S_ijkl = _rearrange_tensor(S_voigt / _VOIGT_COMPLIANCE_FACTORS)

    # Rotate single-crystal compliance tensor to all N orientations at once:
    # S'_nijkl = R_nia R_njb R_nkc R_nld S_abcd
    S_rotated = np.einsum(
        'nia,njb,nkc,nld,abcd->nijkl',
        rot_matrices, rot_matrices, rot_matrices, rot_matrices,
        S_ijkl,
        optimize=True
    )

    S_weighted = np.einsum('n,nijkl->ijkl', weights, S_rotated)

    # Restore the Voigt factors before inverting back to stiffness
    Sij_reuss = _tensor_in_voigt(S_weighted) * _VOIGT_COMPLIANCE_FACTORS

    return np.linalg.inv(Sij_reuss)


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
        If shapes are incompatible, any fraction is negative, the
        fractions sum to zero, or elastic_tensors is not a 3-D
        array of (6, 6) matrices.
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
    total = volume_fractions.sum()
    if total <= 0:
        raise ValueError("Volume fractions must sum to a positive value")
    if not np.isclose(total, 1.0):
        warnings.warn("Volume fractions do not add up to 1, normalising...")
        volume_fractions = volume_fractions / total
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
    TypeError
        If elastic_tensor is not a NumPy array or ODF is not a
        pandas DataFrame.
    ValueError
        If elastic_tensor is not (6, 6), ODF has fewer than 4
        columns, any weight is negative, or the weights sum to zero.
    """
    if not isinstance(elastic_tensor, np.ndarray):
        raise TypeError("elastic_tensor must be a NumPy array")
    if elastic_tensor.shape != (6, 6):
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
    total = weights.sum()
    if total <= 0:
        raise ValueError("ODF volume fractions must sum to a positive value")
    if not np.isclose(total, 1.0):
        weights = weights / total
    return weights


# End of file
