# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: decompose.py                                            #
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
        "others": None
    }

    for symmetry_class, _ in decomposed_elements.items():

        if symmetry_class != "others":

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
    """Calculate the percentage of isotropy and anisotropy
    of the elastic tensor and the percentage represented
    by each of the different symmetry classes.

    Parameters
    ----------
    decomposition : dict
        a Python dictionary with the decomposed elastic tensors

    Returns
    -------
    dict
        a Python dictionary with the percentages
    """

    percentages = {}

    # estimate the sum of the norm of all elastic vectors
    suma = (np.linalg.norm(tensor_to_vector(decomposition['isotropic'])) +
            np.linalg.norm(tensor_to_vector(decomposition['hexagonal'])) +
            np.linalg.norm(tensor_to_vector(decomposition['tetragonal'])) +
            np.linalg.norm(tensor_to_vector(decomposition['orthorhombic'])) +
            np.linalg.norm(tensor_to_vector(decomposition['monoclinic'])) +
            np.linalg.norm(tensor_to_vector(decomposition['others'])))

    # estimate percentages
    percentages['isotropic'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['isotropic'])) / suma, decimals=2)
    percentages['anisotropic'] = np.around(100 - percentages['isotropic'], decimals=2)
    percentages['hexagonal'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['hexagonal'])) / suma, decimals=2)
    percentages['tetragonal'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['tetragonal'])) / suma, decimals=2)
    percentages['orthorhombic'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['orthorhombic'])) / suma, decimals=2)
    percentages['monoclinic'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['monoclinic'])) / suma, decimals=2)
    percentages['others'] = np.around(100 * np.linalg.norm(tensor_to_vector(decomposition['others'])) / suma, decimals=2)

    return percentages

# End of file
