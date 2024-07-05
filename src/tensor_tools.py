# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: tensor_tools.py                                          #
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
from scipy.spatial.transform import Rotation as R


####################################################################
# The following functions, starting with an underscore, are for
# internal use only, i.e. not intended to be used directly by
# the user.
####################################################################

def _symmetrise_tensor(tensor: np.ndarray) -> np.ndarray:
    """Symmetrizes a tensor.

    Parameters
    ----------
    tensor : numpy.ndarray
        The input tensor of shape (n, n).

    Returns
    -------
    numpy.ndarray
        The symmetrized tensor.
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a Numpy array.")

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input must be a 2D square matrix (n x n).")

    ctensor = tensor.copy()
    np.fill_diagonal(ctensor, 0)

    return tensor + ctensor.T


def _rearrange_tensor(Cij: np.ndarray) -> np.ndarray:
    """Rearrange a 6x6 (rank 2, dimension 6) elastic tensor into
    a 3x3x3x3 (rank 4, dimension 3) elastic tensor according to
    Voigt notation. This rearranging improves tensor operations
    while maintaining the original information.

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    Returns
    -------
    numpy.ndarray
        The 3x3x3x3 elastic tensor corresponding to the input
        tensor.

    Raises
    ------
    ValueError
        If Cij is not a 6x6 symmetric NumPy array.

    Notes
    -----
    The function uses a dictionary, `voigt_notation`, to map the
    indices from the 6x6 tensor to the 3x3x3x3 tensor in Voigt
    notation. The resulting tensor, Cijkl, is calculated by
    rearranging elements from the input tensor, Cij, according to
    the mapping.
    """

    # Check if Cij is a 6x6 symmetric matrix
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij should be a 6x6 NumPy array.")

    if not np.allclose(Cij, Cij.T):
        raise ValueError("Cij should be symmetric.")

    voigt_notation = {0: 0, 11: 1, 22: 2, 12: 3, 21: 3, 2: 4, 20: 4, 1: 5, 10: 5}

    Cijkl = np.zeros((3, 3, 3, 3))

    for L in range(3):
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    Cijkl[i, j, k, L] = Cij[
                        voigt_notation[10 * i + j], voigt_notation[10 * k + L]
                    ]

    return Cijkl


def _tensor_in_voigt(C_ijkl: np.ndarray) -> np.ndarray:
    """Convert the 3x3x3x3 (rank 4, dimension 3) elastic tensor
    into a 6x6 (rank 2, dimension 6) elastic tensor according to
    Voigt notation.

    Parameters
    ----------
    C_ijkl : numpy.ndarray
        The 3x3x3x3 elastic tensor to be converted. It should
        be a 4D NumPy array of shape (3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    Raises
    ------
    ValueError
        If C_ijkl is not a 4D NumPy array of shape (3, 3, 3, 3).

    Notes
    -----
    The function maps the elements from the 3x3x3x3 elastic
    tensor (C_ijkl) to the 6x6 elastic tensor (C_ij) using the Voigt
    notation convention.
    """

    # check input
    if not isinstance(C_ijkl, np.ndarray) or C_ijkl.shape != (3, 3, 3, 3):
        raise ValueError("C_ijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    C_ij = np.zeros((6, 6))

    # Divide by 2 because symmetrization will double the elastic
    # contants in the main diagonal
    C_ij[0, 0] = 0.5 * C_ijkl[0, 0, 0, 0]
    C_ij[1, 1] = 0.5 * C_ijkl[1, 1, 1, 1]
    C_ij[2, 2] = 0.5 * C_ijkl[2, 2, 2, 2]
    C_ij[3, 3] = 0.5 * C_ijkl[1, 2, 1, 2]
    C_ij[4, 4] = 0.5 * C_ijkl[0, 2, 0, 2]
    C_ij[5, 5] = 0.5 * C_ijkl[0, 1, 0, 1]

    C_ij[0, 1] = C_ijkl[0, 0, 1, 1]
    C_ij[0, 2] = C_ijkl[0, 0, 2, 2]
    C_ij[0, 3] = C_ijkl[0, 0, 1, 2]
    C_ij[0, 4] = C_ijkl[0, 0, 0, 2]
    C_ij[0, 5] = C_ijkl[0, 0, 0, 1]

    C_ij[1, 2] = C_ijkl[1, 1, 2, 2]
    C_ij[1, 3] = C_ijkl[1, 1, 1, 2]
    C_ij[1, 4] = C_ijkl[1, 1, 0, 2]
    C_ij[1, 5] = C_ijkl[1, 1, 0, 1]

    C_ij[2, 3] = C_ijkl[2, 2, 1, 2]
    C_ij[2, 4] = C_ijkl[2, 2, 0, 2]
    C_ij[2, 5] = C_ijkl[2, 2, 0, 1]

    C_ij[3, 4] = C_ijkl[1, 2, 0, 2]
    C_ij[3, 5] = C_ijkl[1, 2, 0, 1]

    C_ij[4, 5] = C_ijkl[0, 2, 0, 1]

    return C_ij + C_ij.T


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalizes a vector in 3d cartesian space to lie
    on the unit sphere.

    Parameters
    ----------
    vector : numpy.ndarray, shape (3,)
        a vector in 3d cartesian space

    Returns
    -------
    numpy.ndarray, shape (3,)
        The normalized 3D vector.

    Raises
    ------
    ValueError
        If the input vector does not have a shape of (3,).

    Notes
    -----
    The function normalizes the input vector by dividing each component
    by the vector's magnitude to ensure it lies on the unit sphere.

    Example
    --------
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> normalize_vector(v)
    array([0.26726124, 0.53452248, 0.80178373])
    """
    if vector.shape != (3,):
        raise ValueError("Input vector must have shape (3,).")

    magnitude = np.linalg.norm(vector)

    # Handle the zero vector case
    if magnitude == 0:
        return vector

    return vector / magnitude


def _rotate_Cijkl(
    stiffness_tensor: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a 3x3x3x3 symmetric stiffness tensor using a rotation
    matrix and Einstein summation (numpy.einsum). The operation is
    as follows:

    C'ijkl = Ria x Rjb x Rkc x Rld x Cabcd

    where Cabcd and C'ijkl are the original and the rotated tensor,
    respectively, and R the rotation matrix

    Parameters
    ----------
    stiffness_tensor : numpy.ndarray
        symmetric stiffness tensor

    rotation_matrix : numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    numpy.ndarray
        Rotated 3x3x3x3 symmetric stiffness tensor
    """
    # Ensure the inputs
    assert stiffness_tensor.shape == (3, 3, 3, 3), "Input tensor must be 3x3x3x3"
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"

    rotated_tensor = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        rotation_matrix,
        rotation_matrix,
        rotation_matrix,
        rotation_matrix,
        stiffness_tensor,
    )

    return rotated_tensor

def rotate_stiffness_tensor(
    stiffness_tensor: np.ndarray,
    angle_degrees: float,
    rotation_axis: str = "z"
    ) -> tuple[np.ndarray, np.ndarray]:
    """Rotates a stiffness matrix (Voigt notation) or a
    stiffness tensor around a specified axis. The rotation
    is performed in the right-handed coordinate system taking
    into account the following reference frame:

    TODO

    Parameters
    ----------
    stiffness_tensor : np.ndarray of shape 6x6 or 3x3x3x3
        The original stiffness matrix in Voigt notation 6x6
        or in tensor format (3x3x3x3)

    angle_degrees : float
        The rotation angle in degrees (positive for counterclockwise rotation).

    rotation_axis : str, optional
        The axis around which to rotate:
        "x": Rotate around x-axis (fix x, rotate y and z)
        "y": Rotate around y-axis (fix y, rotate x and z)
        "z": Rotate around z-axis (fix z, rotate x and y)
        Default is "z".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - The rotated 6x6 stiffness matrix (Voigt notation)
        - The rotated 3x3x3x3 stiffness tensor
    """

    # Sanity input checks
    if not isinstance(stiffness_tensor, np.ndarray):
        raise ValueError("Cij should be a NumPy array.")
    
    if not isinstance(rotation_axis, str):
        raise ValueError("rotation_axis should be a string.")

    if stiffness_tensor.shape != (3, 3, 3, 3) or stiffness_tensor.shape != (6, 6):
        raise ValueError(
            "Input stiffness array must be 3x3x3x3 or 6x6 (Voigt notation)"
        )

    # convert 6x6 to 3x3x3x3
    if stiffness_tensor.shape != (6, 6):
        stiffness_tensor = _rearrange_tensor(stiffness_tensor)

    # generate the rotation
    if rotation_axis == "z":
        rotation = R.from_euler("z", angle_degrees, degrees=True)
    elif rotation_axis == "x":
        rotation = R.from_euler("x", angle_degrees, degrees=True)
    elif rotation_axis == "y":
        rotation = R.from_euler("y", angle_degrees, degrees=True)
    else:
        raise ValueError("rotation_axis must be 'x', 'y', or 'z'")

    # rotate tensor
    rotated_Cijkl = _rotate_Cijkl(stiffness_tensor, rotation.as_matrix())

    # get the rotated tensor in Voigt notation
    rotated_Cij = _tensor_in_voigt(rotated_Cijkl)

    return rotated_Cij, rotated_Cijkl


# End of file
