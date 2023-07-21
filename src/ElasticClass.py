# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module
#
# Filename: ElasticClass.py
# Description: TODO
#
# Copyright (c) 2023. 
#
# PyRockWave is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyRockWave is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyRockWave.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Marco A. Lopez-Sanchez, http://orcid.org/0000-0002-0261-9267
# Email: lopezmarco [to be found at] uniovi.es
# Website: https://marcoalopez.github.io/PyRockWave/
# Project Repository: https://github.com/marcoalopez/PyRockWave
#######################################################################

# Import statements
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ElasticProps:
    """A class that encapsulates and calculates various elastic
    properties of materials."""

    mineral_name: str | None
    crystal_system: str | None
    temperature: float  # in °C
    pressure: float     # in GPa
    density: float      # density in g/cm3
    Cij: np.ndarray     # stiffness tensor in GPa
    reference: str

    # to estimate
    Sij: np.ndarray = field(init=False)  # compliance tensor
    K_voigt: float = field(init=False)
    K_reuss: float = field(init=False)
    K_hill: float = field(init=False)
    G_voigt: float = field(init=False)
    G_reuss: float = field(init=False)
    G_hill: float = field(init=False)
    universal_anisotropy: float = field(init=False)
    isotropic_poisson_ratio: float = field(init=False)
    isotropic_avg_vp: float = field(init=False)
    isotropic_avg_vs: float = field(init=False)
    isotropic_avg_vpvs: float = field(init=False)

    def __post_init__(self):
        # Validate crystal system
        valid_crystal_systems = ["Cubic", "cubic",
                                 "Tetragonal", "tetragonal",
                                 "Orthorhombic", "orthorhombic",
                                 "Rhombohedral", "rhombohedral",
                                 "Trigonal", "trigonal",
                                 "Hexagonal", "hexagonal",
                                 "Monoclinic", "monoclinic",
                                 "Triclinic", "triclinic",
                                 None]
        if self.crystal_system not in valid_crystal_systems:
            raise ValueError("Invalid crystal system. Please choose one of the following: "
                             "Cubic, Tetragonal, Orthorhombic, Rhombohedral, Hexagonal, "
                             "Trigonal, Monoclinic, Triclinic, or None")

        # check the symmetry of the elastic tensor
        if not np.allclose(self.Cij, self.Cij.T):
            raise Exception("the elastic tensor is not symmetric!")

        # Calculate the compliance tensor
        self.Sij = np.linalg.inv(self.Cij)

        # unpack the elastic constants to make the equations easier to read
        c11, c22, c33, c44, c55, c66 = np.diag(self.Cij)
        s11, s22, s33, s44, s55, s66 = np.diag(self.Sij)
        c12, c13, c23 = self.Cij[0, 1], self.Cij[0, 2], self.Cij[1, 2]
        s12, s13, s23 = self.Sij[0, 1], self.Sij[0, 2], self.Sij[1, 2]

        # Calculate the bulk modulus Voigt average
        self.K_voigt = 1/9 * ((c11 + c22 + c33) + 2*(c12 + c23 + c13))

        # Calculate the bulk modulus Reuss average
        self.K_reuss = 1 / ((s11 + s22 + s33) + 2*(s12 + s23 + s13))

        # Calculate bulk modulus VRH average
        self.K_hill = (self.K_voigt + self.K_reuss) / 2

        # Calculate the shear modulus Voigt average
        self.G_voigt = 1/15 * ((c11 + c22 + c33) - (c12 + c23 + c13)
                               + 3*(c44 + c55 + c66))

        # Calculate the shear modulus Reuss average
        self.G_reuss = 15 / (4*(s11 + s22 + s33) - 4*(s12 + s23 + s13)
                             + 3*(s44 + s55 + s66))

        # Calculate shear modulus VRH average
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

        # Pugh's ratio
        self.pugh_reuss = self.K_reuss / self.G_reuss
        self.pugh_hill = self.K_hill / self.G_hill
        self.pugh_voigt = self.K_voigt / self.G_voigt

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

    def __repr__(self):
        output = str(type(self)) + "\n"
        output += "\n"
        output += f"Mineral Name: {self.mineral_name}\n"
        output += f"Reference Source: {self.reference}\n"
        output += f"Crystal System: {self.crystal_system}\n"
        output += f"Pressure (GPa): {self.pressure:.1f}\n"
        output += f"Temperature (°C): {self.temperature:.0f}\n"
        output += f"Density (g/cm3): {self.density:.3f}\n"
        output += "\n"
        output += f"Elastic Tensor (Cij) in GPa:\n{self.Cij}\n"
        output += "\n"
        output += "Calculated average properties:\n"
        output += "Bulk Modulus (GPa) → VRH, (Reuss, Voigt)\n"
        output += f"{self.K_hill:.3f}, ({self.K_reuss:.3f}, {self.K_voigt:.3f})\n"
        output += "\n"
        output += "Shear Modulus (GPa) → VRH, (Reuss, Voigt)\n"
        output += f"{self.G_hill:.3f}, ({self.G_reuss:.3f}, {self.G_voigt:.3f})\n"
        output += "\n"
        output += "Isotropic Poisson Ratio → VRH, (Reuss, Voigt)\n"
        output += f"{self.isotropic_poisson_hill:.3f}, ({self.isotropic_poisson_reuss:.3f}, {self.isotropic_poisson_voigt:.3f}))\n"
        output += "\n"
        output += "Pugh's ratio → VRH, (Reuss, Voigt)\n"
        output += f"({self.pugh_hill:.3f}, ({self.pugh_reuss:.3f}, {self.pugh_voigt:.3f})\n"
        output += "\n"
        output += "Anisotropy indexes\n"
        output += f"Universal Elastic Anisotropy: {self.universal_anisotropy:.3f}\n"
        output += f"Kube's Anisotropy Index (proportional): {self.Kube_anisotropy:.3f}\n"
        output += "\n"
        output += "Isotropic seismic properties → VRH, (Reuss, Voigt)\n"
        output += f"Vp (km/s): {self.isotropic_vp_hill:.3f}, ({self.isotropic_vp_reuss:.3f}, {self.isotropic_vp_voigt:.3f})\n"
        output += f"Vs (km/s): {self.isotropic_vs_hill:.3f}, ({self.isotropic_vs_reuss:.3f}, {self.isotropic_vs_voigt:.3f})\n"
        output += f"Vp/Vs: {self.isotropic_vpvs_hill:.3f}, ({self.isotropic_vpvs_reuss:.3f}, {self.isotropic_vpvs_voigt:.3f})\n"

        return output
