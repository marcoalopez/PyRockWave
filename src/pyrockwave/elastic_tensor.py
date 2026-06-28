# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: elastic_tensor.py                                                 #
# Description: Defines the ElasticProps dataclass and supporting functions    #
# for computing and decomposing the elastic properties of materials.          #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2023-present, Marco A. Lopez-Sanchez. All rights reserved.    #
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
from dataclasses import dataclass, field
from typing import Optional

from .decomposition import decompose_Cij, calc_percentages


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


# End of file
