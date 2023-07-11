import numpy as np
import pandas as pd


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
    tuple[float, float, float, float, float]
        Tuple containing Vp0, Vs0, epsilon, delta, and gamma.
    """

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(cij)
    c13 = cij[0, 2]

    # estimate polar speeds
    Vp0 = np.sqrt(c33 / density)
    Vs0 = np.sqrt(c44 / density)

    # estimate Thomsen parameters
    epsilon = (c11 - c33) / (2 * c33)
    delta = ((c13 + c44)**2 - (c33 - c44)**2) / (2 * c33 * (c33 - c44))
    gamma = (c66 - c44) / (2 * c44)

    return Vp0, Vs0, epsilon, delta, gamma


def weak_polar_anisotropy(elastic: elasticClass, mesh: int = 1):
    """ Estimate the speed of body waves as a function
    of propagation direction assuming weak polar anisotropy
    using the Leon Thomsen approach (Thomsen, 1986).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.
    mesh : int, optional
        Density for angles theta and phi, by default 1 which
        means a density of 360 for azimuthal angle and 90 for
        polar angle.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions and calculated
        Vp, Vs1, and Vs2 speeds using the weak polar anisotropy Thomsen model.
    """

    # generate spherical coordinates
    theta = np.arccos(1 - 2 * np.linspace(0, 1, 90/mesh))
    phi = np.linspace(0, 2*np.pi, 360/mesh)

    # extract the elastic tensor and density
    cij = elastic.Cij
    density = elastic.density

    # get Thomsen parameters
    Vp0, Vs0, epsilon, delta, gamma = Thomsen_params(cij, density)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    Vp = Vp0 * (1 + delta * sin_theta**2 * cos_theta**2 + epsilon * sin_theta**4)
    Vsv = Vs0 * (1 + (Vp0 / Vs0)**2 * (epsilon - delta) * sin_theta**2 * cos_theta**2)
    Vsh = Vs0 * (1 + gamma * sin_theta**2)

    # create a meshgrid of all possible combinations
    phi2, theta2, Vp = np.meshgrid(phi, theta, Vp, indexing='ij')
    _, _, Vsv = np.meshgrid(phi, theta, Vsv, indexing='ij')
    _, _, Vsh = np.meshgrid(phi, theta, Vsh, indexing='ij')

    # reshape and store arrays
    data = {'polar_ang': theta2.flatten(),
            'azimuthal_ang': phi2.flatten(),
            'Vp': Vp.flatten(),
            'Vsv': Vsv.flatten(),
            'Vsh': Vsh.flatten()}

    return pd.DataFrame(data)


def polar_anisotropy(elastic: elasticClass, mesh: int = 1):
    """ Estimate the speed of boy waves as a function
    of propagation direction assuming that the material
    have a polar anisotropy using the Don L. Anderson
    approach (Anderson, 1961).

    Parameters
    ----------
    elastic : elasticClass
        The elastic properties of the material.
    mesh : int, optional
        Density for angles theta and phi, by default 1 which
        means a density of 360 for azimuthal angle and 90 for
        polar angle.

    Returns
    -------
    pandas.DataFrame
        Tabular data object containing the propagation directions and calculated
        Vp, Vs1, and Vs2 speeds using the weak polar anisotropy Thomsen model.
    """

    # generate spherical coordinates
    theta = np.arccos(1 - 2 * np.linspace(0, 1, 90/mesh))
    phi = np.linspace(0, 2*np.pi, 360/mesh)

    # unpack some elastic constants for readibility
    c11, _, c33, c44, _, c66 = np.diag(elastic.Cij)
    c13, c14 = elastic.Cij[0, 2], elastic.Cij[0, 3]

    # estimate D value
    first_term = (c33 - c44)**2
    second_term = 2 * (2 * (c13 + c14)**2 - (c33 - c44)*(c11 + c33 - 2*c44)) * np.sin(theta)**2
    third_term = ((c11+c33-2*c44)**2 - 4*(c13+c44)**2) * np.sin(theta)**4
    D = np.sqrt(first_term + second_term + third_term)

    # estimate wavespeeds as a function of propagation polar angle
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    Vp = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 + D))
    Vsv = np.sqrt((1 / (2 * elastic.density)) * (c33 + c44 + (c11 - c33) * sin_theta**2 - D))
    Vsh = np.sqrt((1 / elastic.density) * (c44 * cos_theta**2 + c66 * sin_theta**2))

    # create a meshgrid of all possible combinations
    phi2, theta2, Vp = np.meshgrid(phi, theta, Vp, indexing='ij')
    _, _, Vsv = np.meshgrid(phi, theta, Vsv, indexing='ij')
    _, _, Vsh = np.meshgrid(phi, theta, Vsh, indexing='ij')

    # reshape and store arrays
    data = {'polar_ang': theta2.flatten(),
            'azimuthal_ang': phi2.flatten(),
            'Vp': Vp.flatten(),
            'Vsv': Vsv.flatten(),
            'Vsh': Vsh.flatten()}

    return pd.DataFrame(data)
