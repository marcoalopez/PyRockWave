# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: christoffel.py                                            #
# Description: TODO                                                   #
#                                                                     #
# Copyright (c) 2024                                                  #
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
def christoffel_wave_speeds(
    Cij: np.ndarray, density: float, wavevectors: np.ndarray, type="phase"
):
    """_summary_

    Parameters
    ----------
    Cij : numpy.ndarray
        The 6x6 elastic tensor in Voigt notation.

    density : _type_
        _description_

    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        1D or 2D NumPy array.
        If 1D, shape must be (3,).
        If 2D, shape must be (n, 3).

    type : str, optional
        _description_, by default 'phase'

    Raises
    ------
    ValueError
        If Cij is not a 6x6 symmetric NumPy array.
    TODO
    """

    # Sanity checks on inputs
    # Check if Cij is a 6x6 symmetric matrix
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij should be a 6x6 NumPy array.")
    if not np.allclose(Cij, Cij.T):
        raise ValueError("Cij should be symmetric.")
    # validate wavevectors
    if not isinstance(wavevectors, np.ndarray):
        raise ValueError("wavevectors should be a NumPy array.")
    if wavevectors.ndim == 1 and wavevectors.shape[0] == 3:
        wavevectors = wavevectors.reshape(1, 3)
    elif wavevectors.ndim == 2 and wavevectors.shape[1] == 3:
        pass
    else:
        raise ValueError(
            "wavevectors should be a NumPy array of shape (3,) if 1D or (n, 3) if 2D."
        )

    # rearrange tensor Cij → Cijkl
    Cijkl = _rearrange_tensor(Cij)

    # estimate the normalized Christoffel matrix (M) for
    # every wavevector
    Mij = _christoffel_matrix(wavevectors, Cijkl)
    scaling_factor = 1 / density
    norm_Mij = Mij * scaling_factor

    # estimate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = _calc_eigen(norm_Mij)

    # CALCULATE PHASE VELOCITIES (km/s)
    phase_velocities = calc_phase_velocities(eigenvalues)

    if type == "phase":
        # calculate shear wave splitting

        # return a dataframe with phase velocities

        pass

    else:
        # calculate the derivative of the Christoffel matrix (∇M)
        dMijk = _christoffel_gradient_matrix(wavevectors, Cijkl)

        # calculate the derivative of the Christoffel matrix eigenvalues.
        dλ = _eigenvalue_derivatives(eigenvectors, dMijk)

        # CALCULATE GROUP VELOCITIES (km/s)
        Vs2, Vs1, Vp = calc_group_velocities(
            phase_velocities, eigenvectors, dMijk, wavevectors
        )

        # CALCULATE THE ENHANCEMENT FACTOR
        H = _christoffel_matrix_hessian(Cijkl)
        Hλ = _hessian_eigen(H, dλ)
        A = _calc_enhancement_factor(Hλ)

        # return a dataframe with group velocities

        pass


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
    numpy.ndarray of shape (3, 3, 3, 3).
        The equivalent 3x3x3x3 elastic tensor

    Notes
    -----
    The function uses a dictionary, `voigt_notation`, to map the
    indices from the 6x6 tensor to the 3x3x3x3 tensor in Voigt
    notation. The resulting tensor, Cijkl, is calculated by
    rearranging elements from the input tensor, Cij, according to
    the mapping.
    """

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


def _christoffel_matrix(wavevectors: np.ndarray, Cijkl: np.ndarray) -> np.ndarray:
    """Calculate the Christoffel matrix for a given wave vector
    and elastic tensor Cij.

    Parameters
    ----------
    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray
        The Christoffel matri(x)ces as a 2D NumPy array of shape (n, 3, 3).

    Notes
    -----
    The Christoffel matrix is calculated using the formula
    M = k @ Cijkl @ k, where M is the Christoffel matrix, k is a
    wave vector, and Cijkl is the elastic tensor (stiffness matrix).
    """

    # Validate input parameters
    if not isinstance(Cijkl, np.ndarray) or Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl should be a 4D NumPy array of shape (3, 3, 3, 3).")

    # get the number of wave vectors
    n = wavevectors.shape[0]

    # initialize array (pre-allocate)
    Mij = np.zeros((n, 3, 3))

    for i in range(n):
        Mij[i, :, :] = np.dot(wavevectors[i, :], np.dot(wavevectors[i, :], Cijkl))

    return Mij


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
    Mij : numpy.ndarray
        the Christoffel matri(x)ces as a 2D NumPy array of
        shape (n, 3, 3).

    Returns
    -------
    two numpy.ndarrays
        a (n, 3) array with the eigenvalues of each matrix
        a (n, 3, 3) array with the eigenvectors of each matrix
    """

    # get the number of Christoffel matrices to proccess
    n = Mij.shape[0]

    # preallocate arrays
    eigen_values = np.zeros((n, 3))
    eigen_vectors = np.zeros((n, 3, 3))

    # TO REIMPLEMENT ASSUMING A SHAPE (n, 3, 3).
    for i in range(n):
        eigen_values[i, :], eigen_vectors[i, :, :] = np.linalg.eigh(Mij[i, :, :])

    return eigen_values, eigen_vectors


def calc_phase_velocities(eigenvalues: np.ndarray) -> np.ndarray:
    """Estimate the material's sound velocities of a monochromatic
    plane wave, referred to as the phase velocity, as a function of
    crystal/aggreagte orientation from the Christoffel matrix (M).
    It returns three velocities, one primary (P wave) and two
    secondary (S waves). It is assumed that the Christoffel tensor
    provided is already normalised to the density of the material.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized Christoffel matrix. A
        numpy array of shape (n, 3)

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Each triad contains the three wave velocities [Vs2, Vs1, Vp],
        where Vs2 < Vs1 < Vp.

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

    # get the number of eigen values to proccess
    n = eigenvalues.shape[0]

    # preallocate array
    phase_speeds = np.zeros((n, 3))

    # TO REIMPLEMENT ASSUMING A SHAPE (n, 3)
    for i in range(n):
        phase_speeds[i, :] = np.sign(eigenvalues[i, :]) * np.sqrt(
            np.absolute(eigenvalues[i, :])
        )

    return phase_speeds


def _christoffel_gradient_matrix(
    wavevectors: np.ndarray,
    Cijkl: np.ndarray
) -> np.ndarray:
    """Calculate the derivative of the Christoffel matrix. The
    derivative of the Christoffel matrix is computed using
    the formula (e.g. Jaeken and Cottenier, 2016):

    ∂Mij / ∂q_k = ∑m (Cikmj + Cimkj) *  q_m

    Parameters
    ----------
    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Cijkl : numpy.ndarray
        The elastic tensor as a 4D NumPy array of shape (3, 3, 3, 3).


    Returns
    -------
    numpy.ndarray
        The gradient matrix of the Christoffel matrix with respect
        to x_n, reshaped as a 3D NumPy array of shape (n, 3, 3, 3).

    Notes
    -----
    The gradient matrix gradmat[n, i, j] is computed using the wave
    vector and the elastic tensor Cijkl. The derivative of the
    Christoffel matrix with respect a wave vector is given by the
    summation of the products of q_k (wave vector component)
    and the terms (Cikmj + Cimkj). The final result is reshaped to a
    3D matrix with shape (3, 3, 3).
    """

    # get the number of wave vectors
    n = wavevectors.shape[0]

    # initialize array (pre-allocate)
    delta_Mij = np.zeros((n, 3, 3, 3))

    # calculate the gradient matrices from the Christoffel matrices
    for i in range(n):
        delta_Mij_temp = np.dot(
            wavevectors[i, :], Cijkl + np.transpose(Cijkl, (0, 2, 1, 3))
        )
        delta_Mij[i, :, :, :] = np.transpose(delta_Mij_temp, (1, 0, 2))

    return delta_Mij


def _eigenvalue_derivatives(
    eigenvectors: np.ndarray, gradient_matrix: np.ndarray
) -> np.ndarray:
    """Calculate the derivatives of eigenvectors with respect to
    the gradient matrix.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Array of shape (n, 3, 3) representing three eigenvectors
        for each Christoffel matrix calculated.

    gradient_matrix : numpy.ndarray
        The derivative of the Christoffel matrix, which has a
        shape of (n, 3, 3, 3)

    Returns
    -------
    numpy.ndarray:
        Array of shape (n, 3, 3), where the i-th index
        corresponds to the i-th eigenvector, and each element
        (i, j, k) represents the derivative of the i-th
        eigenvector with respect to the j-th component of
        the k-th eigenvector.
    """

    # get the number of wave vectors
    n = eigenvectors.shape[0]

    # initialize array (pre-allocate)
    eigen_derivatives = np.zeros((n, 3, 3))

    # calculate the derivatives
    for ori in range(n):
        temp = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                temp[i, j] = np.dot(eigenvectors[ori, i], np.dot(gradient_matrix[ori, j], eigenvectors[ori, i]))

        eigen_derivatives[ori] = temp

    return eigen_derivatives


def group_velocities(
    phase_velocities: np.ndarray,
    eigenvalue_gradients: np.ndarray,
    wave_vectors: np.ndarray,
):
    """
    Calculate the group velocities for seismic waves in given
    directions, and their magnitudes and directions.

    Parameters
    ----------
    phase_velocities : np.ndarray of shape (n, 3)
        Phase velocities for n directions and 3 wave
        polarizations (P, S1, S2).

    eigenvalue_gradients : np.ndarray of shape (n, 3, 3)
        Gradients of phase velocities with respect to wavenumber
        for n directions, 3 polarizations, and 3 spatial dimensions.

    wavevectors : numpy.ndarray
        The wave vectors normalized to lie on the unit sphere as a
        Numpy array of shape (n, 3).

    Returns
    -------
    tuple containing:
        group_velocities : np.ndarray of shape (n, 3, 3)
            Group velocities for n directions, 3 wave polarizations,
            and 3 spatial dimensions.
        group_velocity_magnitudes : np.ndarray of shape (n, 3)
            Magnitudes of group velocities for n directions and
            3 wave polarizations.
        group_velocity_directions : np.ndarray of shape (n, 3, 3)
            Unit vectors representing directions of group velocities
            for n directions and 3 wave polarizations.
        power flow angles : np.ndarray of shape (n, 3)
            Power flow angles in radians.

    Notes
    -----
    The group velocity is calculated using the relation:
    v_g = (d/dk) * v_p / (2 * v_p)
    where v_g is the group velocity, d/dk is the derivative with respect to the wave vector,
    and v_p is the phase velocity.
    """

    # Ensure the input arrays have the correct shapes
    # assert phase_velocities.ndim == 2 and phase_velocities.shape[1] == 3, "phase_velocities must be of shape (n, 3)"
    # assert eigenvalue_gradients.ndim == 3 and eigenvalue_gradients.shape[1:] == (3, 3), "eigenvalue_gradients must be of shape (n, 3, 3)"

    # Calculate group velocity matrices
    phase_velocities_reshaped = phase_velocities[:, :, np.newaxis]
    group_velocities = eigenvalue_gradients / (2 * phase_velocities_reshaped)

    # Calculate magnitudes of group velocities
    group_velocity_magnitudes = np.linalg.norm(group_velocities, axis=2)

    # Calculate directions of group velocities (unit vectors)
    epsilon = 1e-10  # Add a small epsilon to avoid division by zero
    group_velocity_directions = group_velocities / (
        group_velocity_magnitudes[:, :, np.newaxis] + epsilon
    )

    # calculate thepower flow angles (in radians) for each wave direction
    # cos_power_flow_angles = np.dot(group_velocity_directions, wave_vectors.T)
    # power_flow_angles = np.arccos(np.clip(cos_power_flow_angles, -1, 1))

    return (
        group_velocities,
        group_velocity_magnitudes,
        group_velocity_directions,
        # power_flow_angles
    )


def calc_spherical_angles(group_directions: np.ndarray) -> np.ndarray:
    """ Calculate spherical angles (polar, azimuthal) for a given
    velocity group directions (3D vectors).

    Parameters
    ----------
    group_directions : np.ndarray of shape (n, 3, 3)
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    
    # get the number of wave vectors
    n = group_directions.shape[0]

    # initialize array (pre-allocate)
    angles = np.zeros((n, 2))

    # calculate the derivatives
    for ori in range(n):
        x, z = group_directions[ori, : 0], group_directions[ori, : 2]

        # handle edge cases for z near ±1
        near_pole


# TODO (UNTESTED!)
def calc_group_velocities(
    phase_velocities: np.ndarray,
    eigenvectors: np.ndarray,
    christoffel_gradient: np.ndarray,
    wave_vectors: np.ndarray
) -> np.ndarray:
    """Calculates the group velocities and power flow angles
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
    phase_velocities : numpy.ndarray
        Phase velocities for a particular direction or directions

    eigenvectors : numpy.ndarray
        Eigenvectors of the Christoffel matrix.

    christoffel_gradient : numpy.ndarray
        Gradient of the Christoffel matrix (∇M)
    """

    pass


def _christoffel_matrix_hessian(Cijkl: np.ndarray) -> np.ndarray:
    """Compute the Hessian of the Christoffel matrix. The Hessian
    of the Christoffel matrix, denoted as hessianmat[i, j, k, L]
    here represents the second partial derivatives of the Christoffel
    matrix Mij with respect to the spatial coordinates q_k and q_m
    Can be calculated directly from the stiffness tensor using the
    formulas (e.g. Jaeken and Cottenier, 2016):

    hessianmat[i, j, k, L] = ∂^2Mij / ∂q_k * ∂q_m
    hessianmat[i, j, k, L] = Cikmj + Cimkj

    Note that the Hessian of the Christoffel matrix is independent
    of q (wavevectors)

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

    # initialize array (pre-allocate)
    hessianmat = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for L in range(3):
                    hessianmat[i, j, k, L] = Cijkl[k, i, j, L] + Cijkl[k, j, i, L]

    return hessianmat


def _hessian_eigen(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    delta_Mij: np.ndarray,
    hess_matrix: np.ndarray,
) -> np.ndarray:
    """Compute the hessian of eigenvalues.

    Hessian[n][i][j] = d^2 lambda_n / dx_i dx_j

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        The eigenvalues of the normalized Christoffel matrices,
        which has a shape of (n, 3)

    eigenvectors : numpy.ndarray
        The eigenvectors of the normalized Christoffel matrices,
        which has a shape of (n, 3, 3)

    delta_Mij : numpy.ndarray
        The derivatives of the Christoffel matrices, which has a
        shape of (n, 3, 3)

    hess_matrix : numpy.ndarray
        The Hessian matrix, which has a shape of (3, 3, 3, 3)

    Returns
    -------
    np.ndarray
        _description_
    """

    # get the number of orientations
    n = eigenvectors.shape[0]

    # initialize array (pre-allocate)
    hessian = np.zeros((n, 3, 3, 3))

    # procedure
    for wavevector in range(n):
        diag = np.zeros((n, 3, 3))  # initialize array

        for j in range(3):
            hessian[j] += np.dot(
                np.dot(hess_matrix, eigenvectors[wavevector, j]), eigenvectors[wavevector, j]
            )

            for i in range(3):
                x = eigenvalues[wavevector, j] - eigenvalues[wavevector, i]
                if abs(x) < 1e-10:
                    diag[i, i] = 0.0
                else:
                    diag[i, i] = 1.0 / x

            pseudoinv = np.dot(np.dot(eigenvectors[wavevector].T, diag), eigenvectors[wavevector])
            deriv_vec = np.dot(delta_Mij[wavevector], eigenvectors[wavevector, j])

            # Take derivative of eigenvectors into account: 2 * (d/dx s_i) * pinv_ij * (d_dy s_j)
            hessian[j] += 2.0 * np.dot(np.dot(deriv_vec, pseudoinv), deriv_vec.T)
    
    return hessian


def _calc_enhancement_factor(Hλ):
    pass


# End of file
