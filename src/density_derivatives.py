def olivine(P, dataset='Mao_MT'):
    """ Returns the corresponding density in g/cm3 of San
    Carlos olivine (Fo87-92) up to ~13 GPa based on
    linear fitting from Mao et al. (2015) and
    Zhang and Bass (2016) data.

    Caveat: they measure the density variation at a
    fixed temperature of:
        - 26°C (300 K): Zhang_RT (R-squared=0.9976)
        - 627°C (900 K) Mao_MT (R-squared=0.9961)
        - 1027°C (1300 K) Zhang_HT (R-squared=0.8772)

    Parameters
    ----------
    P : numeric or array-like
        pressure in GPa
    dataset : string, optional
        the fitting, either Mao_MT, Zhang_HT, Zhang_LT
    """

    if dataset == 'Zhang_RT':
        return 0.0225 * P + 3.341

    elif dataset == 'Mao_MT':
        return 0.0228 * P + 3.279

    elif dataset == 'Zhang_HT':
        return 0.0253 * P + 3.239

    else:
        print("dataset must be 'Zhang_RT', 'Mao_MT' or 'Zhang_HT'")


def omphacite(P):
    """ Returns the corresponding density in g/cm3 of
    Omphacite up to 18 GPa based on polynomial fitting
    (R-squared=1) from Hao et al. (2019) data.

    Caveat: they measure the density variation at RT

    Parameters
    ----------
    P : numeric or array-like
        pressure in GPa
    """
    return -0.0002 * P**2 + 0.0266 * P + 3.34


def diopside(P):
    """ Returns the corresponding density in g/cm3 of
    Diopside up to 14 GPa based on polynomial fitting
    (R-squared=0.9999) from Sang and Bass (2014) data.

    Caveat: they measure the density variation at RT

    Parameters
    ----------
    P : numeric or array-like
        pressure in GPa
    """

    return -0.0003 * P**2 + 0.0279 * P + 3.264


def enstatite(P):
    """ Returns the corresponding density in g/cm3 of
    Diopside up to 10.5 GPa based on polynomial fitting
    (R-squared=1) from Zhang and Bass (2014) data.

    Caveat: they measure the density variation at RT

    Parameters
    ----------
    P : numeric or array-like
        pressure in GPa
    """

    return -0.0005 * P**2 + 0.028 * P + 3.288


def alpha_quartz(P):
    """ Returns the corresponding density in g/cm3 of
    alpha quartz up to 10.2 GPa based on polynomial fitting
    (R-squared=0.9999) from Wang et al. (2006) data.

    Caveat: they measure the density variation at RT

    Parameters
    ----------
    P : numeric or array-like
        pressure in GPa
    """

    return -0.00017 * P**2 + 0.064 * P + 2.648
