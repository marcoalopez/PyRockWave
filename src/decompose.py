###############################################################################
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: decompose.py                                                      #
# Description: TODO                                                           #
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
###############################################################################

# Import statements
import numpy as np
import scipy as sp


# Function definitions
def decompose_Cij(Cij: np.ndarray) -> dict:
    """
    Decomposes an elastic tensor after the formulation set out in
    Browaeys and Chevrot (2004). They propose a decomposition of the
    elastic tensor by representing it as a triclinic elastic vector, X,
    before transforming it via a cascade of projections into a sum of
    vectors belonging to the different symmetry classes.

    Parameters
    ----------
    Cij : np.ndarray, shape(6, 6)
        The 6x6 elastic tensor in Voigt notation (in GPa).

    Raises
    ------
    ValueError
        If Cij is not a 6x6 symmetric NumPy array.

    Returns
    -------
    decomposed_elements : dict
        Dictionary containing constitutive symmetry components as key, value pairs.
    """

    # Check if Cij is a 6x6 symmetric matrix
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij should be a 6x6 NumPy array.")

    if not np.allclose(Cij, Cij.T):
        raise ValueError("Cij should be symmetric.")

    Cij_copy = np.copy(Cij)

    decomposed_elements = {
        "isotropic": None,
        "hexagonal": None,
        "tetragonal": None,
        "orthorhombic": None,
        "monoclinic": None,
        "remainder": None
    }

    for symmetry_class, _ in decomposed_elements.items():

        if symmetry_class != "remainder":

            X_total = tensor_to_vector(Cij_copy)
        
            # compute the vector X on a specific symmetry subspace
            M = orthogonal_projector(symmetry_class)
            X_symmetry_class = np.dot(M, X_total)  # X_h = M*X

            C_symmetry_class = np.around(vector_to_tensor(X_symmetry_class), decimals=2)

            # store and subtract
            decomposed_elements[symmetry_class] = C_symmetry_class
            Cij_copy -= C_symmetry_class
        
        else:
            C_symmetry_class = Cij - (decomposed_elements['isotropic'] +
                                      decomposed_elements['hexagonal'] +
                                      decomposed_elements['tetragonal'] +
                                      decomposed_elements['orthorhombic'] +
                                      decomposed_elements['monoclinic'])
            decomposed_elements[symmetry_class] = C_symmetry_class 

    return decomposed_elements


def tensor_to_vector(Cij: np.ndarray) -> np.ndarray:
    """
    Convert the 6x6 elastic tensor Cij to a 21-component elastic vector
    as described in Equation 2.2 of Browaeys and Chevrot (2004).

    Parameters
    ----------
    Cij : numpy.ndarray, shape(6, 6)
        The 6x6 elastic tensor in Voigt notation.

    Returns
    -------
    X : numpy.ndarray
        Elastic vector representation of the elastic tensor in GPa.

    """
    rt2 = np.sqrt(2)
    rt22 = rt2 * 2
    X = np.zeros(21)

    # Diagonal components: C11 , C22 , C33
    X[:3] = Cij[0, 0], Cij[1, 1], Cij[2, 2]

    # Off-diagonal components: √2*C23, √2*C13, √2*C12
    X[3:6] = rt2 * Cij[1, 2], rt2 * Cij[0, 2], rt2 * Cij[0, 1]

    # Pure shear components: 2*C44, 2*C55, 2*C66
    X[6:9] = 2 * Cij[3, 3], 2 * Cij[4, 4], 2 * Cij[5, 5]

    # Shear-normal components: 2*C14, 2*C25, 2*C36
    #                          2*C34, 2*C15, 2*C26
    #                          2*C24, 2*C35, 2*C16
    X[9:12] = 2 * Cij[0, 3], 2 * Cij[1, 4], 2 * Cij[2, 5]
    X[12:15] = 2 * Cij[2, 3], 2 * Cij[0, 4], 2 * Cij[1, 5]
    X[15:18] = 2 * Cij[1, 3], 2 * Cij[2, 4], 2 * Cij[0, 5]

    # Others: 2*√2*C56 , 2*√2*C46 , 2*√2*C45
    X[18:21] = rt22 * Cij[4, 5], rt22 * Cij[3, 5], rt22 * Cij[3, 4]

    return X


def vector_to_tensor(X: np.ndarray) -> np.ndarray:
    """
    Convert an elastic vector, X, of shape (21,) to an elastic
    tensor Cij (6x6) as described in Equation 2.2 of Browaeys
    and Chevrot (2004).

    Parameters
    ----------
    X : np.ndarray, shape(21,)
        Elastic vector representation of the elastic tensor Xi
        in GPa.

    Returns
    -------
    Cij : np.ndarray, shape(6, 6)
        Elastic tensor for the material, in GPa.

    Raises
    ------
    ValueError
        If the length of the input vector X is not (21,).
    """

    if X.shape != (21,):
        raise ValueError("Input vector X must have a shape of (21,)")

    rt2 = np.sqrt(2)
    rt22 = rt2 * 2

    # set equivalence Xi → Cij
    # Diagonal components
    C11, C22, C33 = X[0], X[1], X[2]
    # Off-diagonal components
    C23 = X[3] / rt2
    C13 = X[4] / rt2
    C12 = X[5] / rt2
    # Pure shear components
    C44 = X[6] / 2
    C55 = X[7] / 2
    C66 = X[8] / 2
    # Shear-normal components  
    C14 = X[9] / 2
    C25 = X[10] / 2
    C36 = X[11] / 2
    C34 = X[12] / 2
    C15 = X[13] / 2
    C26 = X[14] / 2
    C24 = X[15] / 2
    C35 = X[16] / 2
    C16 = X[17] / 2
    # Others:
    C56 = X[18] / rt22
    C46 = X[19] / rt22
    C45 = X[20] / rt22    

    Cij = np.array(
        [[ C11, C12, C13, C14, C15, C16],
         [ C12, C22, C23, C24, C25, C26],
         [ C13, C23, C33, C34, C35, C36],
         [ C14, C24, C34, C44, C45, C46],
         [ C15, C25, C35, C45, C55, C56],
         [ C16, C26, C36, C46, C56, C66]])

    return Cij

def orthogonal_projector(symmetry_class: str) -> np.ndarray:
    """
    General projector that generates a matrix M in the 21D vectorial
    space described by the orthonormal basis given in Table 1 of
    Browaeys and Chevrot (2004). See also Appendix A in Browaeys
    and Chevrot (2004).

    Browaeys, J.T., Chevrot, S., 2004. Decomposition of the
    elastic tensor and geophysical applications. Geophysical
    Journal International 159, 667–678.
    https://doi.org/10.1111/j.1365-246X.2004.02415.x


    Parameters
    ----------
    symmetry_class : str
        Name of symmetry class required.

    Returns
    -------
    M : np.ndarray, shape(21, 21)
        Projection matrix for the specified symmetry class.
    """

    rt2 = np.sqrt(2)
    M = np.zeros((21, 21))

    # Projection onto the isotropic space (N_h=2)
    if symmetry_class == "isotropic":
        M[0:3, 0:3] = 3 / 15
        M[0:3, 3:6] = rt2 / 15
        M[0:3, 6:9] = 2 / 15

        M[3:6, 0:3] = rt2 / 15
        M[3:6, 3:6] = 4 / 15
        M[3:6, 6:9] = -rt2 / 15

        M[6:9, 0:3] = 2 / 15
        M[6:9, 3:6] = -rt2 / 15
        M[6:9, 6:9] = 1 / 5

    # Projection onto the hexagonal space (N_h=5)
    elif symmetry_class == "hexagonal":
        M[0:2, 0:2] = 3 / 8
        M[0:2, 5] = M[5, 0:2] = 1 / (4 * rt2)
        M[0:2, 8] = M[8, 0:2] = 1 / 4
        M[2, 2] = 1.0
        M[3:5, 3:5] = M[6:8, 6:8] = M[8, 8] = 1 / 2
        M[5, 5] = 3 / 4
        M[5, 8] = M[8, 5] = -1 / (2 * rt2)

    # Projection onto the tetragonal space (N_h=6)
    elif symmetry_class == "tetragonal":
        M[2, 2] = M[5, 5] = M[8, 8] = 1.0
        M[0:2, 0:2] = M[3:5, 3:5] = M[6:8, 6:8] = 1 / 2

    # Projection onto the orthorhombic space (N_h=9)
    elif symmetry_class == "orthorhombic":
        np.fill_diagonal(M, 1)
        M[9:, 9:] = 0

    # Projection onto the monoclinic space (N_h=13)
    elif symmetry_class == "monoclinic":
        np.fill_diagonal(M, 1)
        M[:, 9:11] = M[:, 12:14] = M[:, 15:17] = M[:, 18:20] = 0

    else:
        print('symmetry class not valid')

    return M


def calc_percentages(decomposition: dict) -> dict:
    """Calculate the percentage of isotropy, anisotropy,
    and symmetry classes of the elastic tensor.

    Parameters
    ----------
    decomposition : dict
        A dictionary with the decomposed elastic tensors

    Returns
    -------
    dict
        A dictionary with the calculated percentages
    """

    percentages = {}

    tensor_classes = ['isotropic', 'hexagonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'remainder']

    # estimate the sum of the norm of all elastic vectors
    sum_norms = np.sum([np.linalg.norm(tensor_to_vector(decomposition[tensor_class])) for tensor_class in tensor_classes])

    for tensor_class in tensor_classes:
        norm = np.linalg.norm(tensor_to_vector(decomposition[tensor_class]))
        percentages[tensor_class] = np.around(100 * norm / sum_norms, decimals=2)

    percentages['anisotropic'] = np.around(100 - percentages['isotropic'], decimals=2)

    return percentages


def euclidean_distance(A, B, squared=False):
    """Euclidean distance between matrices.

    Parameters
    ----------
    A : numpy.ndarray
        First matrix, at least 2D ndarray.
    B : numpy.ndarray
        The second matrix, same dimensions as A
    squared : bool, default False
        Return squared distance.

    Returns
    -------
    float
        Euclidean distance between matrices A and B.
    """
    d = np.linalg.norm(A - B, ord='fro', axis=(-2, -1))

    return d ** 2 if squared else d


def log_euclidean_distance(A, B):
    """Compute the log-Euclidean distance between two
    positive definite matrices.

    Parameters
    ----------
    A : numpy.ndarray
        The first positive definite matrix.
    B : numpy.ndarray
        The second positive definite matrix.

    Returns
    -------
    float
        The log-Euclidean distance between matrices A and B.
    """
    log_A = sp.linalg.logm(A)
    log_B = sp.linalg.logm(B)
    diff = log_A - log_B
    norm = np.linalg.norm(diff, 'fro')  # Frobenius norm
    return norm

def log_euclidean_distance2(A, B, squared=False):
    """Compute the log-Euclidean distance between two
    positive definite matrices.

    Parameters
    ----------
    A : numpy.ndarray
        The first positive definite matrix.
    B : numpy.ndarray
        The second positive definite matrix.
    squared : bool, default False
        Return squared distance.

    Returns
    -------
    float
        The log-Euclidean distance between matrices A and B.
    """
    log_A = sp.linalg.logm(A)
    log_B = sp.linalg.logm(B)

    return euclidean_distance(log_A, log_B, squared=squared)

# End of file
