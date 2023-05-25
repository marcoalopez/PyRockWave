import numpy as np


def alpha_quartz(P=1e-5):
    """
    Returns the corresponding elastic tensor in GPa as a
    function of pressure (in GPa) based on degree 2 polynomial
    fitting from Wang et al. (2006) data.

    caveats: the fitting for the C44 term is worse constrained,
    R-squared = 0.93. Room temperature.

    Parameters:
    P (numeric or array-like): pressure in GPa

    Returns:
    numpy.ndarray: elastic tensor Cij
    """

    if P > 10.2:
        raise ValueError('The pressure provided is out of the safe range of the model (>10.2 GPa)')

    # Polynomial coefficients
    coeffs = {
        'C11': [-0.0017, 4.1846, 85.509],
        'C33': [-0.299, 14.182, 104.4],
        'C12': [-0.182, 7.26, 6.1394],
        'C13': [-0.2042, 6.8847, 12.928],
        'C14': [0.2053, -3.9378, 17.139],
        'C44': [-0.0916, 1.6144, 58.808]
    }

    # independent terms
    C11 = np.polyval(coeffs['C11'], P)  # R-squared=0.9999
    C33 = np.polyval(coeffs['C33'], P)  # R-squared=0.9977
    C12 = np.polyval(coeffs['C12'], P)  # R-squared=0.9984
    C13 = np.polyval(coeffs['C13'], P)  # R-squared=0.9992
    C14 = np.polyval(coeffs['C14'], P)  # R-squared=0.9931
    C44 = np.polyval(coeffs['C44'], P)  # R-squared=0.9274!! see warning

    # dependent terms
    C66 = 0.5 * (C11 - C12)
    C22, C55, C23, C24, C56 = C11, C44, C13, -C14, C14

    Cij = np.array([[C11, C12, C13, C14, 0.0, 0.0],
                    [C12, C22, C23, C24, 0.0, 0.0],
                    [C13, C23, C33, 0.0, 0.0, 0.0],
                    [C14, C24, 0.0, C44, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, C55, C56],
                    [0.0, 0.0, 0.0, 0.0, C56, C66]])

    return np.around(Cij, decimals=2)


def forsterite_HT(P=1e-5):
    """
    Returns the corresponding elastic tensor in GPa as a
    function of pressure (in GPa) based on degree 2 polynomial
    fitting from Zhang and Bass (2016) data.

    caveat: Fixed temperature at

    Parameters:
    P (numeric or array-like): pressure in GPa

    Returns:
    numpy.ndarray: elastic tensor Cij
    """

    if P > 12.8:
        raise ValueError('The pressure provided is out of the safe range of the model (>12.8 GPa)')

    # Polynomial coefficients
    coeffs = {
        'C11': [-0.0568, 7.9003, 268.47],
        'C22': [-0.1069, 5.5317, 174],
        'C33': [0.0351, 4.3771, 201],
        'C44': [-0.0363, 1.8989, 54],
        'C55': [1.204, 67],
        'C66': [-0.0219, 1.6859, 66],
        'C12': [-0.0581, 3.3446, 67],
        'C13': [-0.055, 2.7464, 63],
        'C23': [-0.0486, 3.4657, 65],
    }

    # independent terms
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

    return np.around(Cij, decimals=2)