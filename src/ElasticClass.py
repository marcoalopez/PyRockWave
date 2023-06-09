# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ElasticProps:
    """A class that encapsulates and calculates various elastic
    properties of materials."""

    mineral_name: str
    crystal_system: str
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
                                 "Triclinic", "triclinic"]
        if self.crystal_system not in valid_crystal_systems:
            raise ValueError("Invalid crystal system. Please choose one of the following: "
                             "Cubic, Tetragonal, Orthorhombic, Rhombohedral, Hexagonal, "
                             "Trigonal, Monoclinic, Triclinic.")

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
                                       + 5*np.log(self.G_voigt / self.G_reuss)**2)

        # Calculate the isotropic average Poisson ratio
        self.isotropic_poisson_ratio = ((3*self.K_hill - 2*self.G_hill)
                                        / (6*self.K_hill + 2*self.G_hill))

        # Pugh's ratio
        self.pugh = self.K_hill / self.G_hill

        # calculate the isotropic average Vp
        Vp = np.sqrt((self.K_hill + 4/3 * self.G_hill) / self.density)
        self.isotropic_avg_vp = np.around(Vp, decimals=4)

        # calculate the isotropic average Vs
        Vs = np.sqrt(self.G_hill / self.density)
        self.isotropic_avg_vs = np.around(Vs, decimals=4)

        # calculate the isotropic average Vp/Vs
        self.isotropic_avg_vpvs = np.around(Vp / Vs, decimals=4)

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
        output += f"Stiffness Tensor (Cij) in GPa:\n{self.Cij}\n"
        output += "\n"
        output += "Calculated average properties:\n"
        output += "Bulk Modulus averages (GPa)\n"
        output += f"Upper bound (Voigt) = {self.K_voigt:.3f}\n"
        output += f"Average (Hill--VRH) = {self.K_hill:.3f}\n"
        output += f"Lower bound (Reuss) = {self.K_reuss:.3f}\n"
        output += "Shear Modulus averages (GPa)\n"
        output += f"Upper bound (Voigt) = = {self.G_voigt:.3f}\n"
        output += f"Average (Hill--VRH)  = {self.G_hill:.3f}\n"
        output += f"Lower bound (Reuss) = {self.G_reuss:.3f}\n"
        output += "Isotropic Average Poisson Ratio\n"
        output += f"Average (Hill--VRH)  = {self.isotropic_poisson_ratio:.3f}\n"
        output += "Pugh's ratio\n"
        output += f"Average (Hill--VRH)  = {self.pugh:.3f}\n"
        output += "\n"
        output += "Anisotropy indexes\n"
        output += f"Universal Elastic Anisotropy: {self.universal_anisotropy:.3f}\n"
        output += f"Kube's Anisotropy Index (proportional): {self.Kube_anisotropy:.3f}\n"
        output += "\n"
        output += "Seismic properties (Hill averages)\n"
        output += f"Isotropic Average Vp (km/s): {self.isotropic_avg_vp:.3f}\n"
        output += f"Isotropic Average Vs (km/s): {self.isotropic_avg_vs:.3f}\n"
        output += f"Isotropic Average Vp/Vs: {self.isotropic_avg_vpvs:.3f}\n"

        return output
