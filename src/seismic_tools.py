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


# Function definitions
def Thomsen_params(cij: np.ndarray, density: float):
    """Estimate the Thomsen paramenters for
    weak polar anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material.
    density : float
        The density of the material.

    Returns
    -------
    [float, float, float, float, float]
        Tuple/List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(cij)
    c13 = cij[0, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c44 / density)

    # estimate Thomsen dimensionless parameters
    epsilon = (c11 - c33) / (2 * c33)
    delta = ((c13 + c44)**2 - (c33 - c44)**2) / (2 * c33 * (c33 - c44))
    gamma = (c66 - c44) / (2 * c44)

    return Vp0, Vs0, epsilon, delta, gamma


def Tsvankin_params(cij: np.ndarray, density: float):
    """Estimate the Tsvankin paramenters for
    azimuthal orthotropic anisotropy.

    Parameters
    ----------
    cij : numpy.ndarray
        The elastic stiffness tensor of the material.
    density : float
        The density of the material.

    Returns
    -------
    [float, float, (float, float, float, float, float, float, float)]
        Tuple/List containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, c22, c33, c44, c55, c66 = np.diag(cij)
    c12, c13, c23 = cij[0, 1], cij[0, 2],  cij[1, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c55 / density)

    # estimate Tsvankin dimensionless parameters
    epsilon1 = (c22 - c33) / (2 * c33)
    delta1 = ((c23 + c44)**2 - (c33 - c44)**2) / (2*c33 * (c33 - c44))
    gamma1 = (c66 - c55) / (2 * c55)
    epsilon2 = (c11 - c33) / (2 * c33)
    delta2 = ((c13 + c55)**2 - (c33 - c55)**2) / (2*c33 * (c33 - c55))
    gamma2 = (c66 - c44) / (2 * c44)
    delta3 = (c12 + c66)**2 - (c11 - c66)**2 / (2*c11 * (c11 - c66))

    return Vp0, Vs0, (epsilon1, delta1, gamma1, epsilon2, delta2, gamma2, delta3)


def weak_polar_anisotropy(elastic):
    """Estimate the speed of body waves in a material as a function
    of propagation direction assuming that the material have a
    weak polar anisotropy using the Thomsen approach (Thomsen, 1986).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using the weak polar
        anisotropy model.
    """

    # generate equispaced spherical coordinates
    azimuths, polar = c.equispaced_S2_grid(n=80_000, hemisphere='upper')

    # get Thomsen parameters
    Vp0, Vs0, epsilon, delta, gamma = Thomsen_params(elastic.Cij, elastic.density)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = Vp0 * (1 + delta * sin_theta**2 * cos_theta**2 + epsilon * sin_theta**4)
    Vsv = Vs0 * (1 + (Vp0 / Vs0)**2 * (epsilon - delta) * sin_theta**2 * cos_theta**2)
    Vsh = Vs0 * (1 + gamma * sin_theta**2)

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp,
            'Vsv': Vsv,
            'Vsh': Vsh}

    return pd.DataFrame(data)


def polar_anisotropy(elastic):
    """Estimate the speed of body waves in a material as a function
    of propagation direction assuming that the material have a
    polar anisotropy using the Anderson approach (Anderson, 1961).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions
        and calculated Vp, Vs1, and Vs2 speeds using a polar
        anisotropy model.
    """

    # generate equispaced spherical coordinates
    azimuths, polar = c.equispaced_S2_grid(n=80_000, hemisphere='upper')

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(elastic.Cij)
    c13 = elastic.Cij[0, 2]

    # estimate D value
    first_term = (c33 - c44)**2
    second_term = 2 * (2 * (c13 + c44)**2 - (c33 - c44)*(c11 + c33 - 2*c44)) * np.sin(polar)**2
    third_term = ((c11+c33-2*c44)**2 - 4*(c13+c44)**2) * np.sin(polar)**4
    D = np.sqrt(first_term + second_term + third_term)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(polar)
    cos_theta = np.cos(polar)
    Vp = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 + D))
    Vsv = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 - D))
    Vsh = np.sqrt((1 / elastic.density) * (c44 * cos_theta**2 + c66 * sin_theta**2))

    # reshape and store arrays
    data = {'polar_ang': polar,
            'azimuthal_ang': azimuths,
            'Vp': Vp,
            'Vsv': Vsv,
            'Vsh': Vsh}

    return pd.DataFrame(data)


def orthotropic_azimuthal_anisotropy(elastic):
    """The simplest realistic case of azimuthal anisotropy is that of
    orthorhombic anisotropy (a.k.a. orthotropic).

    Parameters
    ----------
    elastic : _type_
        _description_
    """
    # TODO
    pass


def estimate_wave_speeds(wave_vectors, density, Cijkl):
    """TODO

    Parameters
    ----------
    wave_vectors : _type_
        _description_
    density : _type_
        _description_
    Cijkl : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    scaling_factor = 1 / density

    # estimate the normalized Christoffel matrix (M) for every
    # wavevector
    M = _christoffel_matrix(wave_vectors, Cijkl)
    M = M * scaling_factor

    # estimate the eigenvalues and polarizations of M
    eigen_values, eigen_vectors = _calc_eigen(M)

    # CALCULATE PHASE VELOCITIES (km/s)
    Vs2, Vs1, Vp = calc_phase_velocities(M)

    # CALCULATE GROUP VELOCITIES (km/s) AND POWERFLOW ANGLES
    # calculate the derivative of the Christoffel matrix (∇M)
    dM = _christoffel_matrix_gradient(wave_vectors, Cijkl)

    # calculate the derivative of the Christoffel matrix eigenvalues.

    # estimate group velocities Vs, position and powerflow angles

    # CALCULATE THE ENHANCEMENT FACTOR
    # estimate the Hessian of the Christoffel matrix (H(M))
    hessian_M = _christoffel_matrix_hessian(M)

    # estimate the Eigenvalues of H(M)

    pass


####################################################################
# The following functions, starting with an underscore, are for
# internal use only, i.e. not intended to be used directly by
# the user.
####################################################################

def _symmetrise_tensor(tensor):
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


def _rearrange_tensor(C_ij):
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


def _tensor_in_voigt(C_ijkl):
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


def _normalize_vector(vector):
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


def _christoffel_matrix(wave_vector, Cijkl):
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


def _christoffel_matrix_gradient(wave_vector, Cijkl):
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


def _christoffel_matrix_hessian(Cijkl):
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


def _calc_eigen(M):
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


def _eigenvector_derivatives(eigenvectors, gradient_matrix):
    """Calculate the derivatives of eigenvectors with respect to
    a derivative of the original matrix.

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


def _rotate_vector_to_plane(vector, azimuth_angle, polar_angle):
    """
    Rotate a 3-vector into a plane.

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


def calculate_polarisation_angle(polar_angle,
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


def calc_phase_velocities(M):
    """Estimate the material's sound velocities of a monochromatic
    plane wave, referred to as the phase velocity, from the
    Christoffel matrix (M) . It returns three velocities,
    one primary (P wave) and two secondary (S waves). It is
    assumed that the Christoffel tensor provided is already
    normalised to the density of the material.

    Parameters
    ----------
    M : numpy.ndarray
        The Christoffel matrix as a 3x3 NumPy array.

    Returns
    -------
    numpy.ndarray
        Three wave velocities [Vs2, Vs1, Vp], where Vp > Vs1 > Vs2.

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
    eigen_values, _ = _calc_eigen(M)

    return np.sign(eigen_values) * np.sqrt(np.absolute(eigen_values))


def calc_group_velocities(phase_velocities,
                          eigenvectors,
                          M_derivative,
                          wave_vector):
    """_summary_

    Parameters
    ----------
    phase_velocities : _type_
        _description_
    eigenvectors : _type_
        _description_
    dM : _type_
        _description_
    """

    velocity_group = np.zeros((3, 3))
    group_abs = np.zeros(3)
    group_dir = np.zeros((3, 3))

    # estimate the derivative of eigenvectors
    eigen_derivatives = _eigenvector_derivatives(eigenvectors, M_derivative)

    # estimate group velocities and...TODO
    for i in range(3):
        for j in range(3):
            velocity_group[i, j] = M_derivative[i, j] / (2 * phase_velocities[i])
        group_abs[i] = np.linalg.norm(velocity_group[i])
        group_dir[i] = velocity_group[i] / group_abs[i]

    # estimate the powerflow angle
    powerflow_angle = np.arccos(np.dot(group_dir, wave_vector))

    return velocity_group, powerflow_angle

# End of file
