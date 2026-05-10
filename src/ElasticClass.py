#=============================================================================#
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: ElasticClass.py                                                   #
# Description: Defines the ElasticProps dataclass and supporting functions    #
# for computing and decomposing the elastic properties of materials.          #
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
# =============================================================================#


# Import statements
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Class definitions
@dataclass
class ElasticProps:
    """
    Encapsulates elastic properties of a crystalline material at a given
    pressure and temperature.

    Computes Voigt, Reuss, and Hill averages of the bulk and shear moduli,
    isotropic wave speeds and ratios, anisotropy indices, and a Browaeys &
    Chevrot (2004) symmetry-class decomposition from the stiffness tensor and
    density supplied at construction time.

    Parameters
    ----------
    temperature : float
        Temperature in degrees Celsius.
    pressure : float
        Pressure in GPa.
    density : float
        Density in g/cm³.  Must be strictly positive.
    Cij : np.ndarray, shape (6, 6)
        Elastic stiffness tensor in Voigt notation (GPa).  Must be symmetric.
    mineral_name : str, optional
        Name of the mineral or phase.
    crystal_system : str, optional
        Crystal system (case-insensitive).  Accepted values: cubic, tetragonal,
        orthorhombic, rhombohedral, trigonal, hexagonal, monoclinic, triclinic.
    rock_type : str, optional
        Rock-type descriptor (e.g. "mantle peridotite").
    reference : str, optional
        Bibliographic reference for the elastic data.

    Attributes
    ----------
    Sij : np.ndarray, shape (6, 6)
        Compliance tensor (GPa⁻¹), computed as the inverse of Cij.
    K_voigt, K_reuss, K_hill : float
        Bulk modulus Voigt, Reuss, and Hill averages (GPa).
    G_voigt, G_reuss, G_hill : float
        Shear modulus Voigt, Reuss, and Hill averages (GPa).
    universal_anisotropy : float
        Universal elastic anisotropy index (Ranganathan & Ostoja-Starzewski 2008).
    Kube_anisotropy : float
        Log-Euclidean anisotropy index (Kube & de Jong 2016).
    isotropic_poisson_voigt, isotropic_poisson_reuss, isotropic_poisson_hill : float
        Isotropic Poisson's ratio for each averaging scheme.
    isotropic_vp_voigt, isotropic_vp_reuss, isotropic_vp_hill : float
        Isotropic P-wave velocity for each averaging scheme (km/s).
    isotropic_vs_voigt, isotropic_vs_reuss, isotropic_vs_hill : float
        Isotropic S-wave velocity for each averaging scheme (km/s).
    isotropic_vpvs_voigt, isotropic_vpvs_reuss, isotropic_vpvs_hill : float
        Isotropic Vp/Vs ratio for each averaging scheme.
    elastic : pd.DataFrame
        Summary table of bulk modulus, shear modulus, and Poisson's ratio
        (columns) for each averaging scheme (rows).
    wavespeeds : pd.DataFrame
        Summary table of Vp, Vs, and Vp/Vs (columns) for each averaging
        scheme (rows).
    decompose : dict[str, np.ndarray]
        Browaeys & Chevrot (2004) symmetry-class component tensors keyed by
        class name: ``"isotropic"``, ``"hexagonal"``, ``"tetragonal"``,
        ``"orthorhombic"``, ``"monoclinic"``, ``"others"``.
    percent : dict[str, float]
        Percentage contribution of each symmetry class (same keys as
        ``decompose``) plus ``"anisotropic"`` (100 − isotropic percentage).
    """

    # dataclass compulsory fields
    temperature: float  # in °C
    pressure: float     # in GPa
    density: float      # density in g/cm3
    Cij: np.ndarray     # stiffness tensor in GPa

    # dataclass optional fields
    mineral_name: Optional[str] = None
    crystal_system: Optional[str] = None
    rock_type: Optional[str] = None
    reference: Optional[str] = None

    # fields estimated internally by the dataclass
    Sij: np.ndarray = field(init=False)  # compliance tensor
    K_voigt: float = field(init=False)
    K_reuss: float = field(init=False)
    K_hill: float = field(init=False)
    G_voigt: float = field(init=False)
    G_reuss: float = field(init=False)
    G_hill: float = field(init=False)
    universal_anisotropy: float = field(init=False)
    Kube_anisotropy: float = field(init=False)
    isotropic_poisson_voigt: float = field(init=False)
    isotropic_poisson_reuss: float = field(init=False)
    isotropic_poisson_hill: float = field(init=False)
    isotropic_vp_voigt: float = field(init=False)
    isotropic_vp_reuss: float = field(init=False)
    isotropic_vp_hill: float = field(init=False)
    isotropic_vs_voigt: float = field(init=False)
    isotropic_vs_reuss: float = field(init=False)
    isotropic_vs_hill: float = field(init=False)
    isotropic_vpvs_voigt: float = field(init=False)
    isotropic_vpvs_reuss: float = field(init=False)
    isotropic_vpvs_hill: float = field(init=False)
    elastic: pd.DataFrame = field(init=False)
    wavespeeds: pd.DataFrame = field(init=False)
    decompose: dict[str, np.ndarray] = field(init=False)
    percent: dict[str, float] = field(init=False)

    def __post_init__(self):
        # Validate Cij shape and type
        if not isinstance(self.Cij, np.ndarray) or self.Cij.shape != (6, 6):
            raise ValueError("Cij must be a (6, 6) NumPy array.")

        # Validate density
        if self.density <= 0:
            raise ValueError("density must be a positive number.")

        # Validate crystal system (case-insensitive)
        valid_crystal_systems = {
            "cubic", "tetragonal", "orthorhombic", "rhombohedral",
            "trigonal", "hexagonal", "monoclinic", "triclinic"
        }
        if self.crystal_system is not None:
            if self.crystal_system.lower() not in valid_crystal_systems:
                raise ValueError("Invalid crystal system. Please choose one of the following: "
                                 "cubic, tetragonal, orthorhombic, rhombohedral, hexagonal, "
                                 "trigonal, monoclinic, or triclinic (case-insensitive).")

        # Validate symmetry of the elastic tensor
        if not np.allclose(self.Cij, self.Cij.T):
            raise ValueError("The elastic tensor is not symmetric.")

        # Calculate the compliance tensor
        self.Sij = np.linalg.inv(self.Cij)

        # unpack the elastic constants to make the equations easier to read
        c11, c22, c33, c44, c55, c66 = np.diag(self.Cij)
        s11, s22, s33, s44, s55, s66 = np.diag(self.Sij)
        c12, c13, c23 = self.Cij[0, 1], self.Cij[0, 2], self.Cij[1, 2]
        s12, s13, s23 = self.Sij[0, 1], self.Sij[0, 2], self.Sij[1, 2]

        # Calculate the bulk modulus
        self.K_voigt = 1/9 * ((c11 + c22 + c33) + 2*(c12 + c23 + c13)) # Voigt average
        self.K_reuss = 1 / ((s11 + s22 + s33) + 2*(s12 + s23 + s13))   # Reuss average
        self.K_hill = (self.K_voigt + self.K_reuss) / 2                # Hill average

        # Calculate the shear modulus
        self.G_voigt = 1/15 * ((c11 + c22 + c33) - (c12 + c23 + c13)
                               + 3*(c44 + c55 + c66))
        self.G_reuss = 15 / (4*(s11 + s22 + s33) - 4*(s12 + s23 + s13)
                             + 3*(s44 + s55 + s66))
        self.G_hill = (self.G_voigt + self.G_reuss) / 2

        # Calculate the Universal elastic anisotropy
        self.universal_anisotropy = (5*(self.G_voigt / self.G_reuss)
                                     + (self.K_voigt / self.K_reuss) - 6)

        # Calculate the Kube's log-Euclidean anisotropy index
        self.Kube_anisotropy = np.sqrt(np.log(self.K_voigt / self.K_reuss)**2
                                       + 5 * np.log(self.G_voigt / self.G_reuss)**2)

        # Calculate the isotropic average Poisson ratio
        self.isotropic_poisson_reuss = ((3*self.K_reuss - 2*self.G_reuss)
                                        / (6*self.K_reuss + 2*self.G_reuss))
        self.isotropic_poisson_hill = ((3*self.K_hill - 2*self.G_hill)
                                       / (6*self.K_hill + 2*self.G_hill))
        self.isotropic_poisson_voigt = ((3*self.K_voigt - 2*self.G_voigt)
                                        / (6*self.K_voigt + 2*self.G_voigt))

        # calculate the isotropic average Vp
        Vp_reuss = np.sqrt((self.K_reuss + 4/3 * self.G_reuss) / self.density)
        Vp_hill = np.sqrt((self.K_hill + 4/3 * self.G_hill) / self.density)
        Vp_voigt = np.sqrt((self.K_voigt + 4/3 * self.G_voigt) / self.density)
        self.isotropic_vp_reuss = np.around(Vp_reuss, decimals=4)
        self.isotropic_vp_hill = np.around(Vp_hill, decimals=4)
        self.isotropic_vp_voigt = np.around(Vp_voigt, decimals=4)

        # calculate the isotropic average Vs
        Vs_reuss = np.sqrt(self.G_reuss / self.density)
        Vs_hill = np.sqrt(self.G_hill / self.density)
        Vs_voigt = np.sqrt(self.G_voigt / self.density)
        self.isotropic_vs_reuss = np.around(Vs_reuss, decimals=4)
        self.isotropic_vs_hill = np.around(Vs_hill, decimals=4)
        self.isotropic_vs_voigt = np.around(Vs_voigt, decimals=4)

        # calculate the isotropic average Vp/Vs
        self.isotropic_vpvs_reuss = np.around(Vp_reuss / Vs_reuss, decimals=4)
        self.isotropic_vpvs_hill = np.around(Vp_hill / Vs_hill, decimals=4)
        self.isotropic_vpvs_voigt = np.around(Vp_voigt / Vs_voigt, decimals=4)

        # table with average elastic properties
        self.elastic = pd.DataFrame({'Unit:GPa': ['Voigt', 'Hill', 'Reuss'],
                                     'Bulk_modulus': [self.K_voigt,
                                                      self.K_hill,
                                                      self.K_reuss],
                                     'Shear_modulus': [self.G_voigt,
                                                       self.G_hill,
                                                       self.G_reuss],
                                     'Poisson_ratio': [self.isotropic_poisson_voigt,
                                                       self.isotropic_poisson_hill,
                                                       self.isotropic_poisson_reuss]})

        # table with isotropic wavespeeds and ratios
        self.wavespeeds = pd.DataFrame({'Unit:km/s': ['Voigt', 'Hill', 'Reuss'],
                                        'Vp': [Vp_voigt, Vp_hill, Vp_reuss],
                                        'Vs': [Vs_voigt, Vs_hill, Vs_reuss],
                                        'Vp/vs': [self.isotropic_vpvs_voigt,
                                                  self.isotropic_vpvs_hill,
                                                  self.isotropic_vpvs_reuss]})
        
        # decompose the elastic tensor into lower symmetriy classes
        self.decompose = decompose_Cij(self.Cij)
        self.percent = calc_percentages(self.decompose)

    def __repr__(self):
        output = str(type(self)) + "\n"
        output += "\n"
        if self.mineral_name is not None:
            output += f"Mineral Name: {self.mineral_name}\n"
        if self.rock_type is not None:
            output += f"Rock type: {self.rock_type}\n"
        output += f"Reference Source: {self.reference}\n"
        output += f"Crystal System: {self.crystal_system}\n"
        output += f"Pressure (GPa): {self.pressure:.1f}\n"
        output += f"Temperature (°C): {self.temperature:.0f}\n"
        output += f"Density (g/cm3): {self.density:.3f}\n"
        output += "\n"
        output += f"Elastic Tensor (Cij) in GPa:\n{self.Cij}\n"
        output += "\n"
        output += "Tensor decomposition (Browaeys & Chevrot approach):\n"
        output += f"    Isotropy = {self.percent['isotropic']:.1f} %\n"
        output += f"    Anisotropy = {self.percent['anisotropic']:.1f} %\n"
        output += "\n"        
        output += "Anisotropy indexes:\n"
        output += f"    Universal Elastic Anisotropy:           {self.universal_anisotropy:.3f}\n"
        output += f"    Kube's Anisotropy Index (proportional): {self.Kube_anisotropy:.3f}\n"
        output += "\n"
        output += "Calculated elastic average properties:\n"
        output += f"{self.elastic.round(3).to_string(index=False)}"
        output += "\n"
        output += "\n"
        output += "Isotropic seismic properties:\n"
        output += f"{self.wavespeeds.round(3).to_string(index=False)}"

        return output


# Function definitions
def decompose_Cij(Cij: np.ndarray) -> dict[str, np.ndarray]:
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
    decomposed_elements : dict[str, np.ndarray]
        Dictionary with keys ``"isotropic"``, ``"hexagonal"``,
        ``"tetragonal"``, ``"orthorhombic"``, ``"monoclinic"``, and
        ``"others"``.  Each value is the corresponding (6, 6) component
        tensor in GPa, ordered from highest to lowest symmetry.
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

    for symmetry_class in decomposed_elements:

        if symmetry_class != "others":
            X_total = tensor_to_vector(Cij_copy)

            # compute the vector X on a specific symmetry subspace
            M = orthogonal_projector(symmetry_class)
            X_symmetry_class = np.dot(M, X_total)  # X_h = M*X

            C_symmetry_class = vector_to_tensor(X_symmetry_class)

            # store and subtract to build the residual for the next step
            decomposed_elements[symmetry_class] = C_symmetry_class
            Cij_copy -= C_symmetry_class

        else:
            # Cij_copy now holds the triclinic residual after all projections
            decomposed_elements["others"] = Cij_copy.copy()

    return decomposed_elements


def tensor_to_vector(Cij: np.ndarray) -> np.ndarray:
    """
    Convert the 6x6 elastic tensor Cij to a 21-component elastic vector
    as described in Equation 2.2 of Browaeys and Chevrot (2004).

    Parameters
    ----------
    Cij : numpy.ndarray, shape(6, 6)
        The 6x6 elastic tensor in Voigt notation.

    Raises
    ------
    ValueError
        If Cij is not a (6, 6) NumPy array.

    Returns
    -------
    X : numpy.ndarray
        Elastic vector representation of the elastic tensor in GPa.

    """
    if not isinstance(Cij, np.ndarray) or Cij.shape != (6, 6):
        raise ValueError("Cij must be a (6, 6) NumPy array.")

    rt2 = np.sqrt(2)
    two_rt2 = rt2 * 2
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
    X[18:21] = two_rt2 * Cij[4, 5], two_rt2 * Cij[3, 5], two_rt2 * Cij[3, 4]

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

    if not isinstance(X, np.ndarray) or X.shape != (21,):
        raise ValueError("Input vector X must be a NumPy array with shape (21,).")

    rt2 = np.sqrt(2)
    two_rt2 = rt2 * 2

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
    C56 = X[18] / two_rt2
    C46 = X[19] / two_rt2
    C45 = X[20] / two_rt2

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
        Name of the symmetry class.  Accepted values: ``"isotropic"``,
        ``"hexagonal"``, ``"tetragonal"``, ``"orthorhombic"``,
        ``"monoclinic"``.

    Raises
    ------
    ValueError
        If ``symmetry_class`` is not one of the accepted values.

    Returns
    -------
    M : np.ndarray, shape(21, 21)
        Orthogonal projection matrix for the specified symmetry class.
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
        raise ValueError(
            f"Invalid symmetry class '{symmetry_class}'. "
            "Expected one of: isotropic, hexagonal, tetragonal, "
            "orthorhombic, monoclinic."
        )

    return M


def calc_percentages(decomposition: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Calculate the percentage contribution of each symmetry class to the
    elastic tensor, following the squared-norm definition of Browaeys and
    Chevrot (2004).

    Parameters
    ----------
    decomposition : dict[str, np.ndarray]
        Dictionary of (6, 6) component tensors as returned by
        :func:`decompose_Cij`.  Must contain the keys ``"isotropic"``,
        ``"hexagonal"``, ``"tetragonal"``, ``"orthorhombic"``,
        ``"monoclinic"``, and ``"others"``.

    Raises
    ------
    ValueError
        If any required key is missing from ``decomposition``.

    Returns
    -------
    percentages : dict[str, float]
        Percentage contribution of each symmetry class, with keys
        ``"isotropic"``, ``"hexagonal"``, ``"tetragonal"``,
        ``"orthorhombic"``, ``"monoclinic"``, ``"others"``, and
        ``"anisotropic"`` (= 100 − isotropic percentage).
    """

    tensor_classes = [
        "isotropic",
        "hexagonal",
        "tetragonal",
        "orthorhombic",
        "monoclinic",
        "others",
    ]

    required = set(tensor_classes)
    if not required.issubset(decomposition):
        missing = required - decomposition.keys()
        raise ValueError(f"decomposition dict is missing keys: {missing}")

    # Compute elastic vectors once; percentages use squared norms so that the
    # decomposition satisfies Parseval's identity (||X||² = Σ||X_class||²).
    sq_norms = {tc: np.linalg.norm(tensor_to_vector(decomposition[tc])) ** 2
                for tc in tensor_classes}
    total_sq_norm = sum(sq_norms.values())

    percentages: dict[str, float] = {
        tc: float(np.around(100 * sq_norms[tc] / total_sq_norm, decimals=2))
        for tc in tensor_classes
    }
    percentages["anisotropic"] = float(np.around(100 - percentages["isotropic"], decimals=2))

    return percentages

# End of file
