import numpy as np
from dataclasses import dataclass, field, asdict


@dataclass
class ElasticTensor:
    """A class that represents an elastic tensor and calculate, store and
    print various derived properties."""

    Cij: np.ndarray  # stiffness tensor in GPa
    density: float  # density in g/cm3
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

    def __post_init__(self):
        """_summary_
        """
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

        # Calculate the isotropic average Poisson ratio
        self.isotropic_poisson_ratio = ((3*self.K_hill - 2*self.G_hill)
                                        / (6*self.K_hill + 2*self.G_hill))

        # calculate the isotropic average Vp
        Vp = np.sqrt((self.K_hill + 4/3 * self.G_hill) / self.density)
        self.isotropic_avg_vp = np.around(Vp, decimals=4)

        # calculate the isotropic average Vs
        Vs = np.sqrt(self.G_hill / self.density)
        self.isotropic_avg_vs = np.around(Vs, decimals=4)

    def print_summary(self):
        print("ElasticTensor instance summary:")
        for k, v in asdict(self).items():
            print(f"{k}: {np.around(v, decimals=3)}")

    def __repr__(self) -> str:
        return ('ElasticTensor class object')
