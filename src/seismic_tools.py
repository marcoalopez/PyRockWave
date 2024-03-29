# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: seismic_tools.py                                          #
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
import pandas as pd
import coordinates as c


####################################################################
# The following functions, starting with an underscore, are for
# internal use only, i.e. not intended to be used directly by
# the user.
####################################################################

def _symmetrise_tensor(tensor: np.ndarray):
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
        raise ValueError("Input must be a numpy array.")

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input must be a 2D square matrix (n x n).")

    ctensor = tensor.copy()
    np.fill_diagonal(ctensor, 0)

    return tensor + ctensor.T


def _rearrange_tensor(C_ij: np.ndarray):
    """Rearrange a 6x6 (rank 2, dimension 6) elastic tensor into
    a 3x3x3x3 (rank 4, dimension 3) elastic tensor according to
    Voigt notation. This rearranging improves tensor operations
    while maintaining the original information.

    Parameters
    ----------
    C_ij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    Returns
    -------
    numpy.ndarray
        The 3x3x3x3 elastic tensor corresponding to the input
        tensor.

    Raises
    ------
    ValueError
        If C_ij is not a 6x6 symmetric NumPy array.

    Notes
    -----
    The function uses a dictionary, `voigt_notation`, to map the
    indices from the 6x6 tensor to the 3x3x3x3 tensor in Voigt
    notation. The resulting tensor, C_ijkl, is calculated by
    rearranging elements from the input tensor, C_ij, according to
    the mapping.
    """

    # Check if C_ij is a 6x6 symmetric matrix
    if not isinstance(C_ij, np.ndarray) or C_ij.shape != (6, 6):
        raise ValueError("C_ij should be a 6x6 NumPy array.")

    if not np.allclose(C_ij, C_ij.T):
        raise ValueError("C_ij should be symmetric.")

    voigt_notation = {0: 0, 11: 1, 22: 2, 12: 3, 21: 3, 2: 4, 20: 4, 1: 5, 10: 5}

    C_ijkl = np.zeros((3, 3, 3, 3))

    for L in range(3):
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    C_ijkl[i, j, k, L] = C_ij[voigt_notation[10 * i + j],
                                              voigt_notation[10 * k + L]]

    return C_ijkl


def _tensor_in_voigt(C_ijkl: np.ndarray):
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


def _normalize_vector(vector: np.ndarray):
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


def _christoffel_matrix(wave_vector: np.ndarray, Cijkl: np.ndarray):
    """Calculate the Christoffel matrix for a given wave vector
    and elastic tensor Cij.

    Parameters
    ----------
    wave_vector : numpy.ndarray
        The wave vector as a 1D NumPy array of length 3.

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray
        The Christoffel matrix as a 2D NumPy array of shape (3, 3).

    Raises
    ------
    ValueError
        If wave_vector is not a 1D NumPy array of length 3, or
        if Cij is not a 4D NumPy array of shape (3, 3, 3, 3).

    Notes
    -----
    The Christoffel matrix is calculated using the formula
    M = k @ Cijkl @ k, where M is the Christoffel matrix, k is the
    wave vector, and Cijkl is the elastic tensor (stiffness matrix).
    """

    # Validate input parameters
    if not isinstance(wave_vector, np.ndarray) or wave_vector.shape != (3,):
        raise ValueError("wave_vector should be a 1D NumPy array of length 3.")

    if not isinstance(Cijkl, np.ndarray) or Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    # normalize wavevector to lie on unit sphere
    wave_vector = _normalize_vector(wave_vector)

    return np.dot(wave_vector, np.dot(wave_vector, Cijkl))


def _christoffel_matrix_gradient(wave_vector: np.ndarray, Cijkl: np.ndarray):
    """Calculate the derivative of the Christoffel matrix. The
    derivative of the Christoffel matrix is computed using
    the formula (e.g. Jaeken and Cottenier, 2016):

    ∂M_ij / ∂q_k = ∑m (C_ikmj + C_imkj) *  q_m

    Parameters
    ----------
    wave_vector : numpy.ndarray
        The wave vector as a 1D NumPy array of length 3.

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).


    Returns
    -------
    numpy.ndarray
        The gradient matrix of the Christoffel matrix with respect
        to x_n, reshaped as a 3D NumPy array of shape (3, 3, 3).

    Notes
    -----
    The gradient matrix gradmat[n, i, j] is computed using the wave
    vector and the elastic tensor Cijkl. The derivative of the
    Christoffel matrix with respect a wave vector is given by the
    summation of the products of q_k (wave vector component)
    and the terms (C_ikmj + C_imkj). The final result is reshaped to a
    3D matrix with shape (3, 3, 3).
    """

    # Validate input parameters
    if not isinstance(wave_vector, np.ndarray) or wave_vector.shape != (3,):
        raise ValueError("wave_vector should be a 1D NumPy array of length 3.")

    if not isinstance(Cijkl, np.ndarray) or Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    gradmat = np.dot(wave_vector, Cijkl + np.transpose(Cijkl, (0, 2, 1, 3)))

    return np.transpose(gradmat, (1, 0, 2))


def _christoffel_matrix_hessian(Cijkl: np.ndarray):
    """Compute the Hessian of the Christoffel matrix. The Hessian
    of the Christoffel matrix, denoted as hessianmat[i, j, k, L]
    here represents the second partial derivatives of the Christoffel
    matrix M_kl with respect to the spatial coordinates x_i and x_j
    (i, j = 0, 1, 2). ICan be calculated using the formulas (e.g.
    Jaeken and Cottenier, 2016):

    hessianmat[i, j, k, L] = ∂^2M_ij / ∂q_k * ∂q_m
    hessianmat[i, j, k, L] = C_ikmj + C_imkj

    Parameters
    ----------
    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray
        The Hessian of the Christoffel matrix as a
        4D NumPy array of shape (3, 3, 3, 3).

    Notes
    -----
    The function iterates over all combinations of the indices
    i, j, k, and L to compute the corresponding elements of the
    Hessian matrix.
    """

    hessianmat = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for L in range(3):
                    hessianmat[i, j, k, L] = Cijkl[k, i, j, L] + Cijkl[k, j, i, L]

    return hessianmat


def _calc_eigen(M: np.ndarray):
    """Return the eigenvalues and eigenvectors of the Christoffel
    matrix sorted from low to high. The eigenvalues are related to
    primary (P) and secondary (S-fast, S-slow) wave speeds. The
    eigenvectors provide the unsigned polarization directions,
    which are orthogonal to each other. It is assumed that the
    Christoffel tensor provided is already normalised to the
    density of the material.

    Parameters
    ----------
    M : numpy ndarray
        the Christoffel matrix (3x3)

    Returns
    -------
    two numpy ndarrays
        a (,3) array with the eigenvalues
        a (3,3) array with the eigenvectors
    """

    # Check if M is a 3x3 array
    if not isinstance(M, np.ndarray) or M.shape != (3, 3):
        raise ValueError("M should be a 3x3 NumPy array.")

    eigen_values, eigen_vectors = np.linalg.eigh(M)

    return (eigen_values[np.argsort(eigen_values)],
            eigen_vectors.T[np.argsort(eigen_values)])


def _eigenvector_derivatives(eigenvectors: np.ndarray,
                             gradient_matrix: np.ndarray):
    """Calculate the derivatives of eigenvectors with respect to
    the gradient matrix.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape (3, 3) representing three eigenvectors
        of the Christoffel matrix.
    gradient_matrix : numpy.ndarray
        The derivative of the Christoffel matrix, which has a
        shape of (3, 3, 3)

    Returns
    -------
    numpy.ndarray:
        Array of shape (3, 3, 3), where the i-th index
        corresponds to the i-th eigenvector, and each element
        (i, j, k) represents the derivative of the i-th
        eigenvector with respect to the j-th component of
        the k-th eigenvector.
    """

    eigen_derivatives = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            eigen_derivatives[i, j] = np.dot(eigenvectors[i],
                                             np.dot(gradient_matrix[j],
                                                    eigenvectors[i]))

    return eigen_derivatives


def _rotate_vector_to_plane(vector: np.ndarray,
                            azimuth_angle,
                            polar_angle):
    """Rotate a 3-vector into a plane.

    Parameters
    ----------
    vector : numpy ndarray
        A 3-vector (cartesian coordinates)
    azimuth_angle : float
        Azimuth angle in radians, from coordinates x1 towards x2
    polar_angle : float
        Polar angle in radians, from the x1-x2 plane towards x3.

    Returns
    -------
    rotated_vector : array
        Vector within the x-y plane.

    """

    # Create rotation matrices.
    r1 = np.array([[np.cos(azimuth_angle), np.sin(azimuth_angle), 0],
                   [-np.sin(azimuth_angle), np.cos(azimuth_angle), 0],
                   [0, 0, 1]])
    r2 = np.array([[np.cos(polar_angle), 0, -np.sin(polar_angle)],
                   [0, 1, 0],
                   [np.sin(polar_angle), 0, np.cos(polar_angle)]])

    # Rotate the vector.
    rotated_vector = np.dot(np.dot(vector, r1), r2)

    return rotated_vector


def polarisation_angle(polar_angle,
                       azimuth_angle,
                       polarisation_vector):
    """Calculates the projection angle of the polarisation
    vector of the fast shear wave onto the wavefront plane.

    Parameters
    ----------
    polar_angle : float or numpy ndarray
        Polar angle in radians
    azimuth_angle : float or numpy ndarray
        Azimuth angle in radians
    polarisation_vector : float or numpy ndarray
        Polarisation vector of the fast shear wave.

    Returns
    -------
    polarisation_angle : float
        Projection angle of the polarisation vector of
        the fast shear wave onto the wavefront plane.
    """

    # estimate the wavefront vector in cartesian coordinates
    x, y, z = c.sph2cart(azimuth_angle, polar_angle)
    wavefront_vector = _normalize_vector(np.array([x, y, z]))

    # Calculate the normal vector of the wavefront and
    # normalize to unit length
    normal = np.cross(wavefront_vector,
                      np.cross(wavefront_vector,
                               polarisation_vector))
    normal = _normalize_vector(normal)

    # Rotate the normal vector of the wavefront into the x-y plane
    # Create rotation matrices.
    rotated_vector = _rotate_vector_to_plane(normal, azimuth_angle, polar_angle)

    # Calculate the projection angle of the polarisation vector
    # onto the wavefront plane
    angle = np.rad2deg(np.arctan2(rotated_vector[1], rotated_vector[2]))
    angle = angle + 180 if angle < -90 else angle
    angle = angle - 180 if angle > 90 else angle

    return angle

# End of file
