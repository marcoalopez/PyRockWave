# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/
import numpy as np
from ElasticClass import ElasticTensor


def alpha_quartz(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of forsterite as a function of pressure based on a
    polynomial fit from experimental data of Wang et al. (2006)
    [1]

    Caveats
    -------
        - C44 elastic term is worse constrained than the others.
        The fitting has an R-squared value of 0.96
        - The function does not account for temperature effects
        and assumes room temperature.

    Parameters
    ----------
    P : numeric, optional
        Pressure in GPa. Default value is 1e-5 GPa (RT).

    Returns
    -------
    properties : ElasticTensor dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system of alpha quartz ('Trigonal').
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - reference: Reference to the source publication.
        - Other average (seismic) properties

    Examples
    --------
    >>> quartz_props = alpha_quartz(1.0)

    References
    ----------
    [1] Wang, J., Mao, Z., Jiang, F., Duffy, T.S., 2015. Elasticity of
    single-crystal quartz to 10 GPa. Phys Chem Minerals 42, 203–212.
    https://doi.org/10.1007/s00269-014-0711-z
    """

    if (P > 10.2) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 10.2 GPa')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [0.0275, 3.7960, 86.6],
        'C33': [-0.2456, 13.461, 106.4],
        'C12': [-0.1659, 7.043, 6.74],
        'C13': [-0.2184, 7.0753, 12.4],
        'C14': [0.223, -4.1765, 17.8],
        'C44': [0.0224, -0.4466, 3.0089, 58.0]
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9946
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9975
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9984
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9992
    C14 = np.polyval(coeffs['C14'], P)  # R-squared=0.9929
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9579

    # elastic dependent terms
    C66 = 0.5 * (C11 - C12)
    C22, C55, C23, C24, C56 = C11, C44, C13, -C14, C14

    Cij = np.array([[C11, C12, C13, C14, 0.0, 0.0],
                    [C12, C22, C23, C24, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [C14, C24, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, C56],
                    [0.0, 0.0, 0.0, 0.0, C56, C66]])

    # estimate density, R-squared=0.9999
    density = -0.00017 * P**2 + 0.064 * P + 2.648

    properties = ElasticTensor(
        mineral_name='alpha_quartz',
        crystal_system='Trigonal',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1007/s00269-014-0711-z')

    return properties


def forsterite_ZB2016(P=1e-5, type='HT'):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of forsterite as a function of pressure based on a
    polynomial fit from experimental data of Zhang and Bass (2016)
    http://dx.doi.org/10.1002/2016GL069949

    Caveats
    -------
        - No temperature derivative, fixed at 1027°C (1300 K)
        or 26°C (300 K)

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5 (RP)

    type : str, optional
        either 'RT' or 'HT', default 'HT'

    Example
    -------
    density, cijs = forsterite_HT(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if (P > 12.8) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 12.8 GPa')

    # Polynomial coefficients of elastic independent terms
    if type == 'HT':
        coeffs = {
            'C11': [-0.0496, 7.7691, 269],
            'C22': [-0.1069, 5.5317, 174],
            'C33': [0.0351, 4.3771, 201],
            'C44': [-0.0363, 1.8989, 54],
            'C55': [1.204, 67],
            'C66': [-0.0219, 1.6859, 66],
            'C12': [-0.0581, 3.3446, 67],
            'C13': [-0.055, 2.7464, 63],
            'C23': [-0.0486, 3.4657, 65],
        }

    elif type == 'RT':
        pass  # TODO

    else:
        raise ValueError("type must be 'RT' (i.e. room T) or 'HT' (i.e. 1027°C)")

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9960
    C22 = np.polyval(coeffs['C22'], P)  # R-squared=0.9969
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9966
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9916
    C55 = np.polyval(coeffs['C55'], P)  # R-squared=0.9951
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=1.0
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9982
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9891
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.9948

    Cij = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                    [C12, C22, C23, 0.0, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, C66]])

    # estimate density, R-squared=0.8772
    density = 0.0253 * P + 3.239

    return np.around(density, decimals=3), np.around(Cij, decimals=2)


def forsterite_Mao(P=1e-5, T=627):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of forsterite as a function of pressure based on a
    polynomial fit from experimental data of Mao et al. (2015)
    http://dx.doi.org/10.1016/j.epsl.2015.06.045

    Caveats
    -------
        - TODO

    Parameters
    ----------
    P : numeric or array-like, optional
        pressure in GPa, by default 1e-5

    T : numeric, optional
        pressure in °C, by default 627°C

    Example
    -------
    density, cijs = forsterite_MT(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if (P > 13.3) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 13.3 GPa')

    pass


def omphacite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of omphacite as a function of pressure based on a
    polynomial fit from experimental data of Hao et al. (2019)
    http://dx.doi.org/10.1029/2018JB016964

    Caveats
    -------
        - No temperature derivative, room temperature

    Parameters
    ----------
    P : numeric or array-like, optional
        pressure in GPa, by default 1e-5

    Example
    -------
    density, cijs = omphacite(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if P > 18:
        raise ValueError('The pressure is out of the safe range of the model (>18 GPa)')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [-0.0902, 8.109, 231.5],
        'C22': [-0.0015, 5.0018, 201.0],
        'C33': [-0.1048, 7.6197, 253.8],
        'C44': [-0.0003, 0.99, 79.1],
        'C55': [0.0095, 0.9101, 68.9],
        'C66': [0.0094, 2.2866, 74.0],
        'C12': [-0.0499, 4.5027, 84.4],
        'C13': [-0.0034, 3.2224, 76.0],
        'C23': [0.0368, 2.409, 60.0],
        'C15': [0.0309, -0.5988, 7.6],
        'C25': [0.0826, -0.6595, 5.4],
        'C35': [0.0722, -2.2394, 39.8],
        'C46': [-0.0032, 0.0266, 0.1866, 5.9],
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9982
    C22 = np.polyval(coeffs['C22'], P)  # R-squared=0.9972
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9970
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9847
    C55 = np.polyval(coeffs['C55'], P)  # R-squared=0.9926
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9955
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9925
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9916
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.9532
    C15 = np.polyval(coeffs['C15'], P)  # R-squared=0.8325!
    C25 = np.polyval(coeffs['C25'], P)  # R-squared=0.9347!
    C35 = np.polyval(coeffs['C35'], P)  # R-squared=0.9908
    C46 = np.polyval(coeffs['C46'], P)  # R-squared=0.9829

    Cij = np.array([[C11, C12, C13, 0.0, C15, 0.0],
                    [C12, C22, C23, 0.0, C25, 0.0],
                    [C13, C23, C33, 0.0, C35, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, C46],
                    [C15, C25, C35, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, C46, 0.0, C66]])

    # estimate density, R-squared=1
    density = -0.0001 * P**2 + 0.0266 * P + 3.34

    return np.around(density, decimals=3), np.around(Cij, decimals=2)


def diopside(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of diopside as a function of pressure based on a
    polynomial fit from experimental data of Sang and Bass (2014)
    http://dx.doi.org/10.1016/j.pepi.2013.12.011

    Caveats
    -------
        - Room temperature (no temperature derivative)
        - C44, C25, C46 elastic terms with R-squared < 0.95

    Parameters
    ----------
    P : numeric or array-like, optional
        pressure in GPa, by default 1e-5

    Example
    -------
    density, cijs = diopside(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if P > 14:
        raise ValueError('The pressure is out of the safe range of the model (>14 GPa)')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [-0.057, 6.5496, 229],
        'C22': [-0.085, 6.5936, 179],
        'C33': [-0.066, 7.0088, 245.5],
        'C44': [-0.0302, 1.8098, 78.9],
        'C55': [-0.0587, 2.1653, 68.1],
        'C66': [0.0212, 1.8289, 78.2],
        'C12': [-0.1036, 4.7823, 78],
        'C13': [-0.0702, 4.4478, 69.8],
        'C23': [0.0162, 2.9876, 58],
        'C15': [0.0099, -0.1614, 0.1167, 9.9],
        'C25': [8e-5, -1.1773, 6.1],
        'C35': [0.0205, -1.2472, 40.9],
        'C46': [0.1021, -1.5762, 6.6],
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9983
    C22 = np.polyval(coeffs['C22'], P)  # R-squared=0.9964
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9985
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.8827!
    C55 = np.polyval(coeffs['C55'], P)  # R-squared=0.9773
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9520
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9982
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9951
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.8563!
    C15 = np.polyval(coeffs['C15'], P)  # R-squared=0.9924
    C25 = np.polyval(coeffs['C25'], P)  # R-squared=0.7801!
    C35 = np.polyval(coeffs['C35'], P)  # R-squared=0.9665
    C46 = np.polyval(coeffs['C46'], P)  # R-squared=0.6369!

    Cij = np.array([[C11, C12, C13, 0.0, C15, 0.0],
                    [C12, C22, C23, 0.0, C25, 0.0],
                    [C13, C23, C33, 0.0, C35, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, C46],
                    [C15, C25, C35, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, C46, 0.0, C66]])

    # estimate density, R-squared=0.9999
    density = -0.0003 * P**2 + 0.0279 * P + 3.264

    return np.around(density, decimals=3), np.around(Cij, decimals=2)


def enstatite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of orthoenstatite as a function of pressure based
    on a polynomial fit from experimental data of Zhang and
    Bass (2016) http://dx.doi.org/10.1002/2016GL069963

    Caveats
    -------
        - Room temperature (no temperature derivative)

    Parameters
    ----------
    P : numeric or array-like, optional
        pressure in GPa, by default 1e-5

    Example
    -------
    density, cijs = enstatite(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if P > 10.5:
        raise ValueError('The pressure is out of the safe range of the model (>10.5 GPa)')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [-0.3522, 11.84, 232.2],
        'C22': [-0.2049, 9.6744, 175.7],
        'C33': [-0.3367, 12.684, 222.9],
        'C44': [-0.1208, 2.7716, 82.2],
        'C55': [-0.098, 2.4262, 77.3],
        'C66': [-0.1708, 4.0357, 78.1],
        'C12': [-0.1187, 6.5883, 79.5],
        'C13': [-0.2348, 8.0463, 61.2],
        'C23': [-0.1774, 7.8828, 54.1],
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9985
    C22 = np.polyval(coeffs['C22'], P)  # R-squared=0.9966
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9959
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9928
    C55 = np.polyval(coeffs['C55'], P)  # R-squared=0.9790
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9581
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9970
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9912
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.9953

    Cij = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                    [C12, C22, C23, 0.0, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, C66]])

    # estimate density, R-squared=1
    density = -0.0005 * P**2 + 0.028 * P + 3.288

    return np.around(density, decimals=3), np.around(Cij, decimals=2)


def pyrope(P=1e-5, T=476.85):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) of pyrope garnet as a function of pressure based
    on a polynomial fit from experimental data of Lu et al.
    (2013) https://doi.org/10.1016/j.epsl.2012.11.041

    Caveats
    -------
        - TODO

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    T : numeric, optional
        pressure in °C, by default 476.85

    Example
    -------
    density, cijs = enstatite(1.0)

    Returns
    -------
        float: density
        numpy.ndarray: elastic tensor Cij
    """

    if (P > 20) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 20 GPa')

    if (T > 477) | (T < 26):
        raise ValueError('The temperature is out of the safe range of the model: 26 to 477°C')

    # Celsius to K
    T = T + 273.15

    # elastic constant reference values (at 20 GPa and 750 K i.e. 477°C)
    C11_ref = 393.3
    C12_ref = 170.9
    C44_ref = 112.1
    P_ref = 20
    T_ref = 750

    # set P (GPa/K) and T derivatives according to Lu et al. (2013)
    dP_C11 = 6.0
    dP_C12 = 3.5
    dP_C44 = 1.2
    dT_C11 = -20.5/1000
    dT_C12 = -16.3/1000
    dT_C44 = -3.4/1000

    # estimate elastic independent terms
    C11 = C11_ref + (P - P_ref) * dP_C11 + (T - T_ref) * dT_C11
    C12 = C12_ref + (P - P_ref) * dP_C12 + (T - T_ref) * dT_C12
    C44 = C44_ref + (P - P_ref) * dP_C44 + (T - T_ref) * dT_C44

    # elastic dependent terms
    C22, C33 = C11, C11
    C13, C23 = C12, C12
    C55, C66 = C44, C44

    Cij = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                    [C12, C22, C23, 0.0, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, C66]])

    # estimate density, R-squared=1 TODO
    # density = -0.0005 * P**2 + 0.028 * P + 3.288

    return np.around(3.740, decimals=3), np.around(Cij, decimals=2)
