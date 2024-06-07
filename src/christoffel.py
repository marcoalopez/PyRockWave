# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: christoffel.py                                            #
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


# Function definitions
def christoffel_wave_speeds(Cij: np.ndarray,
                            density: float,
                            wavevectors: np.ndarray,
                            type='phase'):
    """_summary_

    Parameters
    ----------
    Cij : _type_
        _description_
    density : _type_
        _description_
    wavevectors : _type_
        _description_
    type : str, optional
        _description_, by default 'phase'
    """
    scaling_factor = 1 / density

    # rearrange tensor Cij → Cijkl
    Cijkl = _rearrange_tensor(Cij)

    # estimate the normalized Christoffel matrix (M) for
    # every wavevector
    Mij = _christoffel_matrix(wavevectors, Cijkl)
    norm_Mij = Mij * scaling_factor

    # estimate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = _calc_eigen(norm_Mij)

    # CALCULATE PHASE VELOCITIES (km/s)
    Vs2, Vs1, Vp = calc_phase_velocities(norm_Mij, eigenvalues)

    if type == 'phase':

        # calculate shear wave splitting

        # return a dataframe with phase velocities

        pass

    else:

        # calculate the derivative of the Christoffel matrix (∇M)
        dMijk = _christoffel_matrix_gradient(wavevectors, Cijkl)

        # calculate the derivative of the Christoffel matrix eigenvalues.
        dλ = _eigenvector_derivatives(eigenvectors, dMijk)

        # CALCULATE GROUP VELOCITIES (km/s)
        Vs2, Vs1, Vp = calc_group_velocities(phase_velocities,
                                             eigenvectors,
                                             dMijk,
                                             wavevectors)

        # CALCULATE THE ENHANCEMENT FACTOR
        H = _christoffel_matrix_hessian(Cijkl)
        Hλ = _hessian_eigen(H, dλ)
        A = _calc_enhancement_factor(Hλ)

        # return a dataframe with group velocities

        pass


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
                    C_ijkl[i, j, k, L] = C_ij[
                        voigt_notation[10 * i + j], voigt_notation[10 * k + L]
                    ]

    return C_ijkl


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


def _christoffel_matrix(wavevector: np.ndarray, Cijkl: np.ndarray):
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
    if not isinstance(wavevector, np.ndarray) or wavevector.shape != (3,):
        raise ValueError("wave_vector should be a 1D NumPy array of length 3.")

    if not isinstance(Cijkl, np.ndarray) or Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    # normalize wavevector to lie on unit sphere
    wave_vector = _normalize_vector(wavevector)

    return np.dot(wave_vector, np.dot(wave_vector, Cijkl))


def _calc_eigen(Mij: np.ndarray):
    """Return the eigenvalues and eigenvectors of the Christoffel
    matrix sorted from low to high. The eigenvalues are related to
    primary (P) and secondary (S-fast, S-slow) wave speeds. The
    eigenvectors provide the unsigned polarization directions,
    which are orthogonal to each other. It is assumed that the
    Christoffel tensor provided is already normalised to the
    density of the material.

    Parameters
    ----------
    Mij : numpy ndarray
        the Christoffel matrix (3x3)

    Returns
    -------
    two numpy ndarrays
        a (,3) array with the eigenvalues
        a (3,3) array with the eigenvectors
    """

    # Check if M is a 3x3 array
    if not isinstance(Mij, np.ndarray) or Mij.shape != (3, 3):
        raise ValueError("M should be a 3x3 NumPy array.")

    eigen_values, eigen_vectors = np.linalg.eigh(Mij)

    return (
        eigen_values[np.argsort(eigen_values)],
        eigen_vectors.T[np.argsort(eigen_values)],
    )


def calc_phase_velocities(eigenvalues: np.ndarray):
    """Estimate the material's sound velocities of a monochromatic
    plane wave, referred to as the phase velocity, as a function of
    crystal/aggreagte orientation from the Christoffel matrix (M).
    It returns three velocities, one primary (P wave) and two
    secondary (S waves). It is assumed that the Christoffel tensor
    provided is already normalised to the density of the material.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized Christoffel matrix

    Returns
    -------
    numpy.ndarray
        Three wave velocities [Vs2, Vs1, Vp], where Vs2 < Vs1 < Vp.

    Notes
    -----
    The function estimates the phase velocities of the material's
    sound waves from the eigenvalues of the Christoffel matrix (M).
    The eigenvalues represent the squared phase velocities, and by
    taking the square root, the actual phase velocities are obtained.
    The output is a 1D NumPy array containing the three velocities,
    Vs2, Vs1, and Vp, sorted in ascending order (Vs2 < Vs1 < Vp).
    Sound waves in nature are never purely monochromatic or planar.
    See calc_group_velocities.
    """

    # Check if eigenvalues is a 3x1 array
    if not isinstance(eigenvalues, np.ndarray) or eigenvalues.shape != (3,):
        raise ValueError("eigenvalues should be a 3x1 NumPy array.")

    return np.sign(eigenvalues) * np.sqrt(np.absolute(eigenvalues))


def _christoffel_matrix_gradient(wavevector: np.ndarray, Cijkl: np.ndarray):
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
    if not isinstance(wavevector, np.ndarray) or wavevector.shape != (3,):
        raise ValueError("wave_vector should be a 1D NumPy array of length 3.")

    if not isinstance(Cijkl, np.ndarray) or Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    gradmat = np.dot(wavevector, Cijkl + np.transpose(Cijkl, (0, 2, 1, 3)))

    return np.transpose(gradmat, (1, 0, 2))


def _eigenvector_derivatives(eigenvectors: np.ndarray, gradient_matrix: np.ndarray):
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


def calc_group_velocities(phase_velocities,
                          eigenvectors,
                          christoffel_gradient,
                          wave_vectors):
    """ Calculates the group velocities and power flow angles
    of seismic waves as a funtion of direction.

    Group velocity is the velocity with which the overall energy of the wave
    propagates. It's calculated as the gradient of the phase velocity with
    respect to the wave vector. The power flow angle is the angle between
    the group velocity (energy direction) and the wave direction.

    This function takes the phase velocities, eigenvectors of the Christoffel matrix,
    derivative of the Christoffel matrix, and propagation direction as input and returns
    the group velocity tensor, group velocity magnitude and direction for each wave mode,
    and the power flow angle.

    Parameters
    ----------
    phase_velocities : array-like
        Phase velocities for a particular direction or directions
    eigenvectors : array-like
        Eigenvectors of the Christoffel matrix.
    christoffel_gradient : array-like
        Gradient of the Christoffel matrix (∇M)
    wave_vectors : array-like
        Direction(s) of wave propagation.
    """

    # Calculate gradients (derivatives) of eigenvalues (v^2)
    # eigenvalue_gradients = _eigenvector_derivatives(eigenvectors, christoffel_gradient)
    eigenvalue_gradients = np.einsum('ij,ijk,ik->ij', eigenvectors, christoffel_gradient, eigenvectors)

    # Group velocity is the gradient of the phase velocity
    velocity_group = eigenvalue_gradients / (2 * phase_velocities[:, np.newaxis])

    # Calculate the magnitude and direction of the group velocity for each mode
    velocity_group_magnitudes = np.linalg.norm(velocity_group, axis=1)
    velocity_group_directions = velocity_group / velocity_group_magnitudes[:, np.newaxis]

    # Calculate power flow angles
    cos_power_flow_angle = np.dot(velocity_group_directions, wave_vectors)
    power_flow_angles = np.arccos(np.around(cos_power_flow_angle, decimals=10))

    return eigenvalue_gradients, velocity_group, power_flow_angles


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


def _hessian_eigen():
    pass


def _calc_enhancement_factor(Hλ):
    pass

# End of file
