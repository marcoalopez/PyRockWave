# This file is part of the Open Database License (ODbL) - version 1.0
#
# Preamble:
# The Mineral Elastic Database is an open database maintained by 
# Marco A. Lopez-Sanchez of the Andalusian Institute of Earth
# Sciences (IACT-CSIC) in Granada (Spain).
#
# License:
# It is licensed under the Open Database License (ODbL) version 1.0.
# Copyright (c) 2024-Present.
#
# You are free:
# 1. To Share: To copy, distribute, and use the database.
# 2. To Create: To produce works from the database.
# 3. To Adapt: To modify, transform, and build upon the database.
#
# As long as you:
# 1. Attribute: You must attribute any public use of the database,
# or works produced from the database, in the manner specified in
# the attribution section below.
# 2. Share-Alike: If you publicly use any adapted version of this
# database or works produced from it, you must also offer that
# adapted database under the ODbL license.
# 3. Keep open: If you redistribute the database, you must make
# the original data available to the public at no cost, without
# any restrictions on the use of the database.
#
# Attribution:
# The attribution requirement under this license is satisfied by
# including the following notice:
# "Mineral Elastic Database by Marco A. Lopez-Sanchez at the
# IACT-CSIC is licensed under the Open Database License (ODbL)  
# version 1.0."
# If you make any changes or adaptations to this database, you must
# indicate so and not imply that the original database is endorsed
# by the Earth Materials Science Laboratory at University of Oviedo.
#
# License Text:
# The full text of the Open Database License (ODbL) version 1.0 is
# available at: https://opendatacommons.org/licenses/odbl/1.0/
# For the avoidance of doubt, this summary is not a license and it
# has no legal value. You should refer to the full text of the ODbL
# for the complete terms and conditions that apply.
#
# Contact:
# If you have any questions or need further clarifications regarding
# this license or the Mineral Elastic Database, you can contact
# Marco A. Lopez-Sanchez at marcoalopez [to be found at] outlook.com
#
# End of License.


# Import statements
import numpy as np

# check 
try:
    from ElasticClass import ElasticProps
except ImportError:
    print("Warning: The ElasticClass.py file should be in the same folder as the database.")

##################################################################
# 1. SILICATES
##################################################################

##################################################################
# 1.1 SILICA GROUP: SiO2

def alpha_quartz(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of alpha quartz
    as a function of pressure based on a polynomial fit from
    experimental data of Wang et al. (2015) [1]

    Caveats
    -------
        - C44 elastic term is worse constrained than the others.
        The fitting has an R-squared value of 0.96
        - The function does not account for temperature effects.
        Room temperature is assumed.

    Parameters
    ----------
    P : numeric, optional
        Pressure in GPa. Default value is 1e-5 GPa (RT).

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

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

    properties = ElasticProps(
        mineral_name='Alpha_quartz',
        crystal_system='Trigonal',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1007/s00269-014-0711-z')

    return properties

##################################################################
# 1.2 OLIVINE GROUP: M2SiO4, where M = Ca, Fe, Mn, Ni, Mg

def forsterite(P=1e-5, type='RT'):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of forsterite
    as a function of pressure based on a polynomial fit from
    experimental data of Zhang and Bass (2016) [1] and Mao et
    al. (2015) [2]

    Caveats
    -------
        - No temperature derivative, fixed at 1027°C (1300 K),
        627°C (900 K), or 26°C (~300 K)
        - Experimental data at 627 °C (Mao et al.) were obtained in
        the pressure range from 4.5 to 13.3 GPa. The elastic properties
        at pressures below 4.5 GPa are extrapolated from the polynomial
        model.

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5 (RP)

    type : str, optional
        either 'RT', 'MT' or 'HT', default 'RT'

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> olivine_props = forsterite(1.0)

    References
    ----------
    [1] Zhang, J.S., Bass, J.D., 2016. Sound velocities of olivine at high
    pressures and temperatures and the composition of Earth’s upper mantle.
    Geophys. Res. Lett. 43, 9611–9618. https://doi.org/10.1002/2016GL069949

    [2] Mao, Z., Fan, D., Lin, J.-F., Yang, J., Tkachev, S.N., Zhuravlev, K.,
    Prakapenka, V.B., 2015. Elasticity of single-crystal olivine at high
    pressures and temperatures. Earth and Planetary Science Letters 426,
    204–215. https://doi.org/10.1016/j.epsl.2015.06.045
    """

    if type == "MT":
        if (P > 13.3) | (P <= 0):
            raise ValueError(
                "The pressure is out of the safe range of the 'MT' model: 0 to 13.3 GPa"
            )
    if type == "RT" or type == "HT":
        if (P > 12.8) | (P <= 0):
            raise ValueError(
                "The pressure is out of the safe range of the models 'RT' or 'HT': 0 to 12.8 GPa"
            )

    # Polynomial coefficients of elastic independent terms
    if type == 'HT':
        coeffs = {
            'C11': [-0.0496, 7.7691, 269], # R-squared=0.9960
            'C22': [-0.1069, 5.5317, 174], # R-squared=0.9969
            'C33': [0.0351, 4.3771, 201],  # R-squared=0.9966
            'C44': [-0.0363, 1.8989, 54],  # R-squared=0.9916
            'C55': [1.204, 67],            # R-squared=0.9951
            'C66': [-0.0219, 1.6859, 66],  # R-squared=1.0
            'C12': [-0.0581, 3.3446, 67],  # R-squared=0.9982
            'C13': [-0.055, 2.7464, 63],   # R-squared=0.9891
            'C23': [-0.0486, 3.4657, 65],  # R-squared=0.9948
        }
        T = 1027
        density = 0.0253 * P + 3.239  # R-squared=0.8772
        reference = 'https://doi.org/10.1002/2016GL069949'
    
    elif type == 'MT':
        coeffs = {
            'C11': [-0.137, 8.1979, 296.02],  # R-squared=0.9990
            'C22': [-0.0145, 5.2479, 179.72], # R-squared=0.9999
            'C33': [-0.0763, 6.1763, 210.01], # R-squared=0.9947
            'C44': [-0.0514, 2.2077, 56.418], # R-squared=0.9999
            'C55': [-0.0455, 1.7866, 71.041], # R-squared=0.9968
            'C66': [0.0037, 1.5318, 71.01],   # R-squared=0.9996
            'C12': [-0.3446, 9.2276, 38.36],  # R-squared=0.9984
            'C13': [-0.1367, 5.2602, 58.15],  # R-squared=0.9994
            'C23': [0.0819, 1.6695, 75.026],  # R-squared=0.9924
        }
        T = 627
        density = -0.0001 * P**2 + 0.0266 * P + 3.34 # R-squared=1
        reference = 'https://doi.org/10.1016/j.epsl.2015.06.045'

    elif type == 'RT':
        coeffs = {
            'C11': [-0.0514, 7.8075, 319.2], # R-squared=0.9969
            'C22': [-0.0930, 5.7684, 195.5], # R-squared=0.9912
            'C33': [-0.0495, 5.7721, 232.7], # R-squared=0.9984
            'C44': [-0.0296, 1.7099, 62.6],  # R-squared=0.9972
            'C55': [-0.0378, 1.6603, 77.5],  # R-squared=0.9916
            'C66': [-0.0660, 2.7941, 75.2],  # R-squared=0.9928
            'C12': [-0.0511, 3.8683, 71.0],  # R-squared=0.9872
            'C13': [-0.0734, 4.8050, 71.0],  # R-squared=0.9891
            'C23': [0.0034, 3.4215, 74.9],   # R-squared=0.9926
        }
        T = 26
        density = -0.0002 * P**2 + 0.0253 * P + 3.3413  # R-squared=1
        reference = 'https://doi.org/10.1002/2016GL069949'

    else:
        raise ValueError("type must be 'RT' (i.e. room T) or 'HT' (i.e. 1027°C)")

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)
    C22 = np.polyval(coeffs['C22'], P)
    C33 = np.polyval(coeffs['C33'], P)
    C44 = np.polyval(coeffs['C44'], P)
    C55 = np.polyval(coeffs['C55'], P)
    C66 = np.polyval(coeffs['C66'], P)
    C12 = np.polyval(coeffs['C12'], P)
    C13 = np.polyval(coeffs['C13'], P)
    C23 = np.polyval(coeffs['C23'], P)

    Cij = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                    [C12, C22, C23, 0.0, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, C66]])

    properties = ElasticProps(
        mineral_name='Forsterite',
        crystal_system='Orthorhombic',
        temperature=T,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference=reference)

    return properties


##################################################################
# 1.3 PYROXENE GROUP: ADSi2O6
# 1.3.1 Clinopyroxene Subgroup (monoclinic)

def omphacite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of omphacite
    as a function of pressure based on a polynomial fit from
    experimental data of Hao et al. (2019) [1]

    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Omph_props = omphacite(1.0)

    References
    ----------
    [1] Hao, M., Zhang, J.S., Pierotti, C.E., Ren, Z., Zhang, D., 2019.
    High‐Pressure Single‐Crystal Elasticity and Thermal Equation of State
    of Omphacite and Their Implications for the Seismic Properties of
    Eclogite in the Earth’s Interior. J. Geophys. Res. Solid Earth 124,
    2368–2377. https://doi.org/10.1029/2018JB016964
    """

    if (P > 18) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 18 GPa')

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

    properties = ElasticProps(
        mineral_name='Omphacite',
        crystal_system='Monoclinic',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1029/2018JB016964')

    return properties


def diopside(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of diopside as
    a function of pressure based on a polynomial fit from
    experimental data of Sang and Bass (2014) [1]

    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.
        - C44, C25, C46 elastic terms with R-squared < 0.95

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Di_props = diopside(1.0)

    References
    ----------
    [1] Sang, L., Bass, J.D., 2014. Single-crystal elasticity of diopside
    to 14GPa by Brillouin scattering. Physics of the Earth and Planetary
    Interiors 228, 75–79. https://doi.org/10.1016/j.pepi.2013.12.011
    """

    if (P > 14) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 14 GPa')

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

    properties = ElasticProps(
        mineral_name='Diopside',
        crystal_system='Monoclinic',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1016/j.pepi.2013.12.011')

    return properties

# 1.3.2 Orthopyroxene Subgroup (monoclinic)

def enstatite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of orthoenstatite
    as a function of pressure based on a polynomial fit from
    experimental data of Zhang and Bass (2016) [1]

    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Ens_props = enstatite(1.0)

    References
    ----------
    [1] Zhang, J.S., Bass, J.D., 2016. Single‐crystal elasticity of natural
    Fe‐bearing orthoenstatite across a high‐pressure phase transition.
    Geophys. Res. Lett. 43, 8473–8481. https://doi.org/10.1002/2016GL069963
    """

    if (P > 10.5) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 10.5 GPa')

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

    properties = ElasticProps(
        mineral_name='Enstatite',
        crystal_system='Orthorhombic',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1002/2016GL069963')

    return properties

##################################################################
# 1.4 GARNET GROUP: X3Z2(SiO4)3, X = Mg, Ca, Fe2+, Mn2+, etc., Z = Al, Fe3+, Cr3+, V3+ etc. 

def pyrope(P=1e-5, T=476.85):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of pyrope garnet
    as a function of pressure based on a polynomial fit from
    experimental data of Lu et al. (2013) [1]

    Caveats
    -------
        - TODO

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    T : numeric, optional
        pressure in °C, by default 476.85

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Grt_props = pyrope(1.0)

    References
    ----------
    [1] Lu, C., Mao, Z., Lin, J.-F., Zhuravlev, K.K., Tkachev, S.N.,
    Prakapenka, V.B., 2013. Elasticity of single-crystal iron-bearing
    pyrope up to 20 GPa and 750 K. Earth and Planetary Science Letters
    361, 134–142. https://doi.org/10.1016/j.epsl.2012.11.041
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

    # estimate density TODO
    density = 3.740

    properties = ElasticProps(
        mineral_name='Pyrope',
        crystal_system='Cubic',
        temperature=T,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1016/j.epsl.2012.11.041')

    return properties

##################################################################
# 1.5 EPIDOTE SUPERGROUP: (A1A2)(M1M2M3)O4[Si2O7][SiO4]O10 

def zoisite():
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of zoisite
    based on experimental data of Mao et al. (2007) [1]

    Caveats
    -------
        - The function does not account for temperature or
        pressure effects and assumes room conditions

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Zo_props = zoisite(1.0)

    References
    ----------
    [1] Mao, Z., Jiang, F., Duffy, T.S., 2007. Single-crystal elasticity
    of zoisite Ca2Al3Si3O12 (OH) by Brillouin scattering. American
    Mineralogist 92, 570–576. https://doi.org/10.2138/am.2007.2329
    """

    # elastic independent terms
    C11 = 279.8
    C22 = 249.2
    C33 = 209.4
    C44 = 51.8
    C55 = 81.4
    C66 = 66.3
    C12 = 94.7
    C13 = 88.7
    C23 = 27.5

    Cij = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                    [C12, C22, C23, 0.0, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, C66]])

    properties = ElasticProps(
        mineral_name='Zoisite',
        crystal_system='Orthorhombic',
        temperature=25,
        pressure=1e-4,
        density=3.343,
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.2138/am.2007.2329')

    return properties


##################################################################
# 1.6 AMPHIBOLE SUPERGROUP: AB2C5((Si,Al,Ti)8O22)(OH,F,Cl,O)2 

def amphiboles(type='Hornblende'):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of different
    types of Ca-Na amphibole based on experimental data of
    Brown and Abramson (2016) [1]

    Caveats
    -------
        - The function does not account for temperature or
        pressure effects and assumes room conditions

    Parameters
    ----------
    type : str
        the type of Ca-Na amphibole

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> amph = amphiboles(type='Pargasite')

    References
    ----------
    [1] Brown, J.M., Abramson, E.H., 2016. Elasticity of calcium and
    calcium-sodium amphiboles. Physics of the Earth and Planetary Interiors
    261, 161–171. https://doi.org/10.1016/j.pepi.2016.10.010
    """

    # elastic independent terms (in GPa) and densities in g/cm3
    if type == 'Hornblende':  # Amph 4 in Table 1
        C11, C22, C33 = 122.8, 189.3, 222.9
        C44, C55, C66 = 71.5, 46.8, 46.2
        C12, C13, C23 = 51.8, 45.9, 62.3
        C15, C25, C35 = -0.7, -7.0, -30.0
        C46 = 5.4
        density = 3.293

    elif type == 'Pargasite':  # Amph 8 in Table 1
        C11, C22, C33 = 141.6, 197.8, 225.4
        C44, C55, C66 = 75.8, 49.9, 51.7
        C12, C13, C23 = 57.1, 49.6, 60.9
        C15, C25, C35 = -0.2, -10.9, -31.4
        C46 = 3.3
        density = 3.163

    elif type == 'Pargasite#2':  # Amph 9 in Table 1
        pass

    elif type == 'Tremolite':  # Amph 5 in Table 1
        C11, C22, C33 = 108.6, 191.6, 230.8
        C44, C55, C66 = 77.0, 50.0, 48.6
        C12, C13, C23 = 48.4, 37.7, 59.2
        C15, C25, C35 = 1.0, -5.6, -29.6
        C46 = 7.9
        density = 3.038

    elif type == 'Richterite':  # Amph 1 in Table 1
        pass

    elif type == 'Kataphorite':  # Amph 2 in Table 1
        pass

    elif type == 'Tschermakite':  # Amph 3 in Table 1
        pass

    elif type == 'Edenite':  # Amph 6 in Table 1
        pass

    elif type == 'Edenite#2':  # Amph 7 in Table 1
        pass

    else:
        raise ValueError("Type must be: 'Hornblende', 'Pargasite',"
                         " 'Pargasite#2','Tremolite', 'Richterite',"
                         " 'Kataphorite', 'Tschermakite', 'Edenite',"
                         " or 'Edenite#2'")

    Cij = np.array([[C11, C12, C13, 0.0, C15, 0.0],
                    [C12, C22, C23, 0.0, C25, 0.0],
                    [C13, C23, C33, 0.0, C35, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, C46],
                    [C15, C25, C35, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, C46, 0.0, C66]])

    properties = ElasticProps(
        mineral_name=type,
        crystal_system='Monoclinic',
        temperature=25,
        pressure=1e-4,
        density=density,
        Cij=Cij,
        reference='https://doi.org/10.1016/j.pepi.2016.10.010')

    return properties

##################################################################
# 1.6 FELDSPAR GROUP: (k,Na,Ca)[(Si,Al)AlSi2]O8

def plagioclase(type='An0'):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of different
    types of plagiclase feldspar based on experimental data of
    Brown et al. (2016) [1]

    Caveats
    -------
        - The function does not account for temperature or
        pressure effects and assumes room conditions

    Parameters
    ----------
    type : str
        the type of plagioclase, either An0, An25, An37, An48,
        An60, An67, An78 or An96. Default An0 or albite

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> plag = plagioclase(type='An37')

    References
    ----------
    [1] Brown, J.M., Angel, R.J., Ross, N.L., 2016. Elasticity of plagioclase
    feldspars. JGR Solid Earth 121, 663–675. https://doi.org/10.1002/2015JB012736
    """

    # elastic independent terms (in GPa) and densities in g/cm3
    if type == 'An0':  # albite
        C11, C22, C33 = 68.3, 184.3, 180.0  # diagonal pure shear
        C44, C55, C66 = 25.0, 26.9, 33.6    # diagonal simple shear
        C12, C13, C23 = 32.2, 30.4, 5.0     # off-diagonal pure shear
        C45, C46, C56 = -2.4, -7.2, 0.6     # off-diagonal simple shear
        C14, C15, C16 = 4.9, -2.3, -0.9     # pure-simple shear relations
        C24, C25, C26 = -4.4, -7.8, -6.4    # ...
        C34, C35, C36 = -9.2, 7.5, -9.4     # ...
        density = 2.623

    elif type == 'An25':  # oligoclase
        C11, C22, C33 = 87.1, 174.9, 166.1
        C44, C55, C66 = 22.9, 29.0, 35.0
        C12, C13, C23 = 43.9, 35.4, 18.0
        C45, C46, C56 = -1.3, -5.2, 0.8
        C14, C15, C16 = 6.1, -0.4, -0.6
        C24, C25, C26 = -5.9, -2.9, -6.5
        C34, C35, C36 = -2.9, 4.6, -10.7
        density = 2.653

    elif type == 'An37':  # andesine
        C11, C22, C33 = 96.2, 189.4, 171.9
        C44, C55, C66 = 23.6, 33.1, 35.5
        C12, C13, C23 = 46.1, 38.4, 15.4
        C45, C46, C56 = -1.1, -4.8, 1.4
        C14, C15, C16 = 5.9, -0.2, -0.4
        C24, C25, C26 = -7.0, -5.1, -6.8
        C34, C35, C36 = 2.2, 7.2, -9.8
        density = 2.666

    elif type == 'An48':  # andesine
        C11, C22, C33 = 104.6, 201.4, 172.8
        C44, C55, C66 = 22.9, 33.0, 35.6
        C12, C13, C23 = 51.5, 43.9, 14.5
        C45, C46, C56 = -1.0, -3.8, 2.1
        C14, C15, C16 = 6.5, 0.1, -0.8
        C24, C25, C26 = -2.4, -4.8, -9.9
        C34, C35, C36 = -0.4, 6.9, -5.7
        density = 2.683

    elif type == 'An60':  # labradorite
        C11, C22, C33 = 109.3, 185.5, 164.1
        C44, C55, C66 = 22.2, 33.1, 36.8
        C12, C13, C23 = 53.1, 42.1, 21.9
        C45, C46, C56 = 0.2, 1.4, 2.8
        C14, C15, C16 = 7.6, 1.2, -7.7
        C24, C25, C26 = -2.9, 0.7, -6.8
        C34, C35, C36 = 0.2, 2.5, 0.7
        density = 2.702

    elif type == 'An67':  # labradorite
        C11, C22, C33 = 120.3, 193.5, 171.9
        C44, C55, C66 = 24.0, 35.5, 37.3
        C12, C13, C23 = 54.4, 40.8, 16.1
        C45, C46, C56 = 0.7, 0.3, 3.2
        C14, C15, C16 = 9.2, 2.3, -10.1
        C24, C25, C26 = 0.9, 3.1, -2.9
        C34, C35, C36 = -0.9, 2.2, -0.3
        density = 2.721

    elif type == 'An78':  # bytownite
        C11, C22, C33 = 120.4, 191.6, 163.7
        C44, C55, C66 = 23.3, 32.8, 35.0
        C12, C13, C23 = 56.6, 49.9, 26.3
        C45, C46, C56 = 0.8, 0.9, 4.5
        C14, C15, C16 = 9.0, 3.2, -3.0
        C24, C25, C26 = 2.1, 5.4, -9.9
        C34, C35, C36 = 1.7, 1.7, -8.1
        density = 2.725

    elif type == 'An96':  # anorthite
        C11, C22, C33 = 132.2, 200.2, 163.9
        C44, C55, C66 = 24.6, 36.6, 36.0
        C12, C13, C23 = 64.0, 55.3, 31.9
        C45, C46, C56 = 3.0, -2.2, 5.2
        C14, C15, C16 = 9.5, 5.1, -10.8
        C24, C25, C26 = 7.5, 3.5, -7.2
        C34, C35, C36 = 6.6, 0.5, 1.6
        density = 2.757

    else:
        raise ValueError("Type must be: 'An0', 'An25', 'An37', 'An48', 'An60', 'An67', 'An78', or 'An96'")

    Cij = np.array([[C11, C12, C13, C14, C15, C16],
                    [C12, C22, C23, C24, C25, C26],
                    [C13, C23, C33, C34, C35, C36],
                    [C14, C24, C34, C44, C45, C46],
                    [C15, C25, C35, C45, C55, C56],
                    [C16, C26, C36, C46, C56, C66]])

    properties = ElasticProps(
        mineral_name='plagioclase_' + type,
        crystal_system='Triclinic',
        temperature=25,
        pressure=1e-4,
        density=density,
        Cij=Cij,
        reference='https://doi.org/10.1002/2015JB012736')

    return properties


##################################################################
# 1.7 PHYLLOSILICATE GROUP
# 1.7.1 Serpentine Subgroup: D3[Si2O5](OH)4, D = Mg, Fe, Ni, Mn, Al, Zn

def antigorite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of antigorite
    as a function of pressure based on a polynomial fit from
    experimental data of Satta et al. (2022) [1]

    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.
        - Some of the experimentally measured elastic constants
        do not show a trend with pressure, but dispersion around
        a value. In these, we decided to use the mean as the
        best model.

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexess

    Examples
    --------
    >>> Atg = antigorite(P=1.0)

    References
    ----------
    [1] Satta, N., Grafulha Morales, L.F., Criniti, G., Kurnosov, A.,
    Boffa Ballaran, T., Speziale, S., Marquardt, K., Capitani, G.C.,
    Marquardt, H., 2022. Single-Crystal Elasticity of Antigorite at
    High Pressures and Seismic Detection of Serpentinized Slabs.
    Geophysical Research Letters 49, e2022GL099411.
    https://doi.org/10.1029/2022GL099411

    """

    if (P > 7.7) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 7.7 GPa')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [-1.1522, 190.6],
        'C33': [-0.6813, 16.628, 85.4],
        'C66': [00.87, 67.5],
        'C12': [0.283, -3.4231, 61.3],
        'C13': [-0.2966, 6.2252, 21.3],
        'C23': [3.839, 19],
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.7037
    C22 = 208.2                         # mean is a better model
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9970
    C44 = 13.5                          # mean is a better model
    C55 = 20.0                          # idem
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9157
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9310
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9660
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.9296
    C15 = 2.9                           # mean is a better model
    C25 = -1.2                          # idem
    C35 = 0.4                           # idem
    C46 = -3.2                          # idem

    Cij = np.array([[C11, C12, C13, 0.0, C15, 0.0],
                    [C12, C22, C23, 0.0, C25, 0.0],
                    [C13, C23, C33, 0.0, C35, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, C46],
                    [C15, C25, C35, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, C46, 0.0, C66]])

    # estimate density, R-squared=0.9967
    density = 0.0356 * P + 2.615

    properties = ElasticProps(
        mineral_name='Antigorite',
        crystal_system='Monoclinic',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1029/2022GL099411')

    return properties

def lizardite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of lizardite
    as a function of pressure based on a polynomial fit from
    experimental data of Tsuchiya et al. (2013) [1]

    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.
        - There is a dramatic change in elastic properties above
        10 GPa, so the returned tensor applies only for pressures below
        this threshold.
        - No experimental measures, atomic first-principles
        - C14 elastic term is worse constrained than the others.
        The fitting has an R-squared value of 0.7

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexess

    Examples
    --------
    >>> Lz = lizardite(P=0.5)

    References
    ----------
    [1] Tsuchiya, J., 2013. A first-principles calculation of the elastic
    and vibrational anomalies of lizardite under pressure. American
    Mineralogist 98, 2046–2052. https://doi.org/10.2138/am.2013.4369

    """

    if (P > 10) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 10 GPa')

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [-0.4758, 5.9191, 212.6],
        'C33': [-0.8992, 22.548, 57.3],
        'C12': [-0.384, 3.72, 73.3],
        'C13': [0.0711, 6.235, 8.5],
        'C14': [-0.0214, 0.2059, 1.3],
        'C44': [-0.1363, 1.0282, 11.6],
        'C66': [-0.0509, 1.1561, 69.7]
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9971
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9986
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9879
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9998
    C14 = np.polyval(coeffs['C14'], P)  # R-squared=0.6951
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9924
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9989

    # elastic dependent terms
    C22, C55, C23, C24, C56 = C11, C44, C13, -C14, C14

    Cij = np.array([[C11, C12, C13, C14, 0.0, 0.0],
                    [C12, C22, C23, C24, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [C14, C24, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, C56],
                    [0.0, 0.0, 0.0, 0.0, C56, C66]])

    # estimate density, R-squared=1
    density = 0.0007 * P**2 + 0.0187 * P + 2.5081

    properties = ElasticProps(
        mineral_name='Lizardite',
        crystal_system='Trigonal',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.2138/am.2013.4369')

    return properties


# 1.7.2 Chlorite group: A5-6T4Z18
# A group of mostly monoclinic (also triclinic or orthorhombic) micaceous
# phyllosilicate minerals

def chlorite(P=1e-5):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of chlorite as
    a function of pressure based on a polynomial fit from first
    principles simulation of Mookherjee and Mainprice (2014) [1]


    Caveats
    -------
        - The function does not account for temperature effects
        and assumes room temperature.
        - The C11, C22 and c44 estimated elastic constant values
        at 1.8 and 4.2 GPa do not follow the trend as the others.
        - Based on first principles simulation

    Parameters
    ----------
    P : numeric, optional
        pressure in GPa, by default 1e-5

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> Chl = chlorite(1.0)

    References
    ----------
    [1] Mookherjee, M., Mainprice, D., 2014. Unusually large shear wave
    anisotropy for chlorite in subduction zone settings. Geophys. Res.
    Lett. 41, 1506–1513. https://doi.org/10.1002/2014GL059334
    """

    if (P > 14) | (P <= 0):
        raise ValueError('The pressure is out of the safe range of the model: 0 to 14 GPa')

    np.set_printoptions(suppress=True)

    # Polynomial coefficients of elastic independent terms
    coeffs = {
        'C11': [0.1674, 0.4206, 197.8],
        'C22': [0.1771, -0.5058, 202.3],
        'C33': [-0.1655, 15.499, 135.1],
        'C44': [0.011, -0.28, 0.4944, 24.5],
        'C55': [0.0154, -0.2914, 0.146, 24.4],
        'C66': [-0.0241, 0.8613, 70.3],
        'C12': [-0.0089, 0.371, -2.2729, 60.7],
        'C13': [-0.0346, 0.765, 0.0758, 21.1],
        'C23': [-0.0564, 1.294, -3.8639, 34.1],
        'C15': [0.0039, -0.0312, 0.1809, 3.3],
        'C25': [0.0024, -0.0082, -0.0769, 0.2],
        'C35': [0.0027, -0.0047, 0.2115, 0.4],
        'C46': [0.004, -0.0626, 0.225, 0.1],
    }

    # elastic independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9829
    C22 = np.polyval(coeffs['C22'], P)  # R-squared=0.9890
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9945
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9973
    C55 = np.polyval(coeffs['C55'], P)  # R-squared=0.9998
    C66 = np.polyval(coeffs['C66'], P)  # R-squared=0.9937
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9985
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9959
    C23 = np.polyval(coeffs['C23'], P)  # R-squared=0.9856
    C15 = np.polyval(coeffs['C15'], P)  # R-squared=0.9991
    C25 = np.polyval(coeffs['C25'], P)  # R-squared=0.9986
    C35 = np.polyval(coeffs['C35'], P)  # R-squared=0.9996
    C46 = np.polyval(coeffs['C46'], P)  # R-squared=0.9829

    Cij = np.array([[C11, C12, C13, 0.0, C15, 0.0],
                    [C12, C22, C23, 0.0, C25, 0.0],
                    [C13, C23, C33, 0.0, C35, 0.0],
                    [0.0, 0.0, 0.0, C44, 0.0, C46],
                    [C15, C25, C35, 0.0, C55, 0.0],
                    [0.0, 0.0, 0.0, C46, 0.0, C66]])

    # estimate density, R-squared=0.9995
    density = -0.0004 * P**2 + 0.0341 * P + 2.534

    properties = ElasticProps(
        mineral_name='Chlorite',
        crystal_system='Monoclinic',
        temperature=25,
        pressure=P,
        density=np.around(density, decimals=3),
        Cij=np.around(Cij, decimals=2),
        reference='https://doi.org/10.1002/2014GL059334')

    return properties


##################################################################
# 1.8 OTHER SILICATES

def kyanite(model='DFT'):
    """
    Returns the corresponding elastic tensor (GPa) and density
    (g/cm3) and other derived elastic properties of kyanite based
    on atomic first-principles as calculated in Winkler et al.
    (2001)[1] (average of Voigt and Reuss models)

    Caveats
    -------
        - The function does not account for temperature or
        pressure effects and assumes room conditions
        - No experimental data to confirm the model
        - Estimates calculated at 0K rather than room temperature

    Parameters
    ----------
    model : str
        whether density functional theory (DFT) or core-shell
        model (THB)

    Returns
    -------
    properties : ElasticProps dataclass
        An object containing the following properties:
        - name: Name of the crystal ('alpha_quartz').
        - crystal_system: Crystal system.
        - temperature: Temperature in degrees Celsius (assumed 25).
        - pressure: Pressure in GPa.
        - density: Density in g/cm3.
        - cijs: 6x6 array representing the elastic tensor.
        - sijs: 6x6 array representing the compliance tensor
        - reference: Reference to the source publication.
        - decompose: the decomposition of the elastic tensor
            into lower symmetriy classes
        - Other average (seismic & elastic) properties
        - Several anisotropy indexes

    Examples
    --------
    >>> ky = kyanite()

    References
    ----------
    [1] Winkler, B., Hytha, M., Warren, M.C., Milman, V., Gale, J.D.,
    Schreuer, J., 2001. Calculation of the elastic constants of the
    Al2SiO5 polymorphs andalusite, sillimanite and kyanite. Zeitschrift
    für Kristallographie - Crystalline Materials 216, 67–70.
    https://doi.org/10.1524/zkri.216.2.67.203366
    """

    # elastic independent terms (in GPa) and densities in g/cm3
    if model == 'DFT':
        C11, C22, C33 = 387, 355, 366  # diagonal pure shear
        C44, C55, C66 = 182, 80, 132   # diagonal simple shear
        C12, C13, C23 = 100, 46, 112   # off-diagonal pure shear
        C45, C46, C56 = -1.5, 0, -6.1  # off-diagonal simple shear
        C14, C15, C16 = -3, 0, -6      # pure-simple shear relations
        C24, C25, C26 = -22, -3, 3     # ...
        C34, C35, C36 = -30, 3, 0      # ...

    elif type == 'THB':
        C11, C22, C33 = 363, 428, 490
        C44, C55, C66 = 203, 117, 110
        C12, C13, C23 = 124, 100, 175
        C45, C46, C56 = -20, -8, 1
        C14, C15, C16 = 8, 14, 12
        C24, C25, C26 = -39, -5, -19
        C34, C35, C36 = -45, -21, -12

    else:
        raise ValueError("Model must be: 'DFT' (default) or 'THB'")

    Cij = np.array([[C11, C12, C13, C14, C15, C16],
                    [C12, C22, C23, C24, C25, C26],
                    [C13, C23, C33, C34, C35, C36],
                    [C14, C24, C34, C44, C45, C46],
                    [C15, C25, C35, C45, C55, C56],
                    [C16, C26, C36, C46, C56, C66]])

    properties = ElasticProps(
        mineral_name='kyanite',
        crystal_system='Triclinic',
        temperature=25,
        pressure=1e-4,
        density=3.67,
        Cij=Cij,
        reference='https://doi.org/10.1524/zkri.216.2.67.203366')

    return properties


if __name__ == '__main__':
    print('Mineral Elastic Database v.2025.2.11')
else:
    print('Mineral Elastic Database v.2025.2.11 imported')

# End of file
