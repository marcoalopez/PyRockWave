# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: layered_media.py                                                  #
# Description: This module calculates seismic reflectivity in layered media.  #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2026, Marco A. Lopez-Sanchez. All rights reserved.            #
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

from .anisotropic_models import tsvankin_params
from .christoffel import _calc_eigen, _christoffel_matrix
from .utils.coordinates import sph2cart
from .utils.tensor_tools import _rearrange_tensor
from .utils.validation import validate_cij


# Function definitions
def snell(
    vp_upper_layer: float | np.ndarray,
    vp_lower_layer: float | np.ndarray,
    vs_upper_layer: float | np.ndarray,
    vs_lower_layer: float | np.ndarray,
    theta1: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the angles of refraction and reflection for an incident
    P-wave in a two-layered system.

    Parameters
    ----------
    vp_upper_layer : float or array-like
        P-wave velocity of upper layer.
    vp_lower_layer : float or array-like
        P-wave velocity of lower layer.
    vs_upper_layer : float or array-like
        S-wave velocity of upper layer.
    vs_lower_layer : float or array-like
        S-wave velocity of lower layer.
    theta1 : float or array-like
        Angle(s) of incidence of P-wave in upper layer in degrees.
        Must be in [0, 90).

    Returns
    -------
    theta2 : float or array-like
        Angle(s) of refraction for P-wave in lower layer in degrees.
    phi1 : float or array-like
        Angle(s) of reflection for S-wave in upper layer in degrees.
    phi2 : float or array-like
        Angle(s) of refraction for S-wave in lower layer in degrees.
    p : float or array-like
        Ray parameter(s).

    Notes
    -----
    At and beyond the critical angle np.arcsin returns nan silently;
    no exception is raised.
    """

    if (np.any(np.asarray(vp_upper_layer) <= 0) or np.any(np.asarray(vp_lower_layer) <= 0)
            or np.any(np.asarray(vs_upper_layer) <= 0) or np.any(np.asarray(vs_lower_layer) <= 0)):
        raise ValueError("All velocities must be positive.")
    if np.any(np.asarray(theta1) < 0) or np.any(np.asarray(theta1) >= 90):
        raise ValueError("theta1 must be in [0, 90).")

    # Convert angles to radians
    theta1 = np.deg2rad(theta1)

    # Calculate ray parameter using Snell's law
    p = np.sin(theta1) / vp_upper_layer

    # Calculate reflection and refraction angles using Snell's law
    # (np.arcsin returns radians; convert back to degrees on output)
    phi1 = np.rad2deg(np.arcsin(p * vs_upper_layer))
    theta2 = np.rad2deg(np.arcsin(p * vp_lower_layer))
    phi2 = np.rad2deg(np.arcsin(p * vs_lower_layer))

    return theta2, phi1, phi2, p


def calc_reflectivity(
    cij_upper_layer: np.ndarray,
    cij_lower_layer: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    incident_angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that computes PP reflectivity in both symmetry
    planes for an interface between two orthorhombic media.

    Extracts Tsvankin parameters from the input stiffness tensors and
    calls :func:`reflectivity`.

    Parameters
    ----------
    cij_upper_layer : np.ndarray
        The 6x6 elastic stiffness tensor of the upper layer in GPa.
    cij_lower_layer : np.ndarray
        The 6x6 elastic stiffness tensor of the lower layer in GPa.
    upper_density_gcm3 : float
        The density of the upper layer in g/cm³.
    lower_density_gcm3 : float
        The density of the lower layer in g/cm³.
    incident_angles_deg : np.ndarray
        Incident angles in degrees.

    Returns
    -------
    Rxz : np.ndarray
        PP reflectivity as a function of angle of incidence in the xz plane.
    Ryz : np.ndarray
        PP reflectivity as a function of angle of incidence in the yz plane.
    """

    # Compute Tsvankin params for both layers
    upper_layer_params = tsvankin_params(cij_upper_layer, upper_density_gcm3)
    lower_layer_params = tsvankin_params(cij_lower_layer, lower_density_gcm3)

    # Delegate to reflectivity, converting angles from degrees to radians
    return reflectivity(
        upper_layer_params,
        lower_layer_params,
        upper_density_gcm3,
        lower_density_gcm3,
        np.deg2rad(incident_angles_deg),
    )


def reflectivity(
    upper_layer_tsvankin_params: np.ndarray,
    lower_layer_tsvankin_params: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    incident_angles_rad: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the P-wave reflectivity in the symmetry plane for interfaces
    between 2 orthorhombic media using the approach of Ruger (1998).

    This is Python reimplementation with improved readability and
    documentation from srb toolbox:
    (https://github.com/StanfordRockPhysics/SRBToolbox)
    
    In particular, a reimplementation of the Rorsym.m file originally
    written by Diana Sava

    Parameters
    ----------
    upper_layer_tsvankin_params : array-like of shape (9,)
        The Tsvankin params of the upper medium (output of
        :func:`tsvankin_params`). At least the first 8 elements
        (through gamma2) are required.
    lower_layer_tsvankin_params : array-like of shape (9,)
        The Tsvankin params of the lower medium (output of
        :func:`tsvankin_params`). At least the first 8 elements
        (through gamma2) are required.
    upper_density_gcm3 : float
        Density of the upper medium in g/cm³.
    lower_density_gcm3 : float
        Density of the lower medium in g/cm³.
    incident_angles_rad : float or array-like
        Incident angle(s) in radians.

    Arrays with Tsvankin params include the following params in this
    specific order (for details refer to :func:`tsvankin_params`):
    -----------------------------------------------------------------
    Vp0 : P-wave vertical velocity of medium (km/s).
    Vs0 : S-wave vertical velocity of medium (km/s).
    epsilon1 (e1) : Epsilon in the first symmetry plane of the
        orthorhombic medium (normal to x).
    delta1 (d1) : Delta in the first symmetry plane of the
        orthorhombic medium (normal to x).
    epsilon2 (e2) : Epsilon in the second symmetry plane of the
        orthorhombic medium (normal to y).
    delta2 (d2) : Delta in the second symmetry plane of the
        orthorhombic medium (normal to y).
    gamma1 (g1) : VTI gamma parameter in the yz symmetry plane,
        (c66 - c55) / (2 * c55).
    gamma2 (g2) : VTI gamma parameter in the xz symmetry plane,
        (c66 - c44) / (2 * c44). Combined with gamma1 to recover the
        vertical shear-wave splitting parameter used in the xz plane.
    delta3 : Present in the array but not used in this reflectivity
        calculation.

    Returns
    -------
    Rxz : np.ndarray
        PP reflectivity as a function of angle of incidence in the xz plane.
    Ryz : np.ndarray
        PP reflectivity as a function of angle of incidence in the yz plane.

    Reference
    ---------
    Ruger, A., 1998, Variation of P-wave reflectivity coefficients
    with offset and azimuth in anisotropic media. Geophysics,
    Vol 63, No 3, p935. https://doi.org/10.1190/1.1444405
    """

    if (len(upper_layer_tsvankin_params) < 8
            or len(lower_layer_tsvankin_params) < 8):
        raise ValueError("Tsvankin parameter arrays must have at least 8 elements.")
    if upper_density_gcm3 <= 0 or lower_density_gcm3 <= 0:
        raise ValueError("Densities must be positive.")

    # Extract Tsvankin params
    # Order: Vp0, Vs0, epsilon1, delta1, epsilon2, delta2, gamma1, gamma2, delta3
    Vp0_up, Vs0_up, e1_up, d1_up, e2_up, d2_up, g1_up, g2_up, *_ = upper_layer_tsvankin_params
    Vp0_low, Vs0_low, e1_low, d1_low, e2_low, d2_low, g1_low, g2_low, *_ = lower_layer_tsvankin_params

    # Ruger's (1998) symmetry-plane equations (23)-(24) use a single shear
    # reference: the vertical S-wave velocity polarised along x2, sqrt(c44/rho),
    # together with the vertical shear-wave splitting parameter
    # gamma_S = (c44 - c55) / (2 * c55) in the xz plane. Vs0 = sqrt(c55/rho)
    # is the x1-polarised velocity, so reconstruct both from the Tsvankin
    # parameters via gamma_S = (gamma1 - gamma2) / (1 + 2*gamma2) and
    # sqrt(c44/rho) = Vs0 * sqrt(1 + 2*gamma_S).
    gS_up = (g1_up - g2_up) / (1 + 2 * g2_up)
    gS_low = (g1_low - g2_low) / (1 + 2 * g2_low)
    Vs2_up = Vs0_up * np.sqrt(1 + 2 * gS_up)
    Vs2_low = Vs0_low * np.sqrt(1 + 2 * gS_low)

    # Calculate impedance for both media
    Z1 = upper_density_gcm3 * Vp0_up
    Z2 = lower_density_gcm3 * Vp0_low

    # Calculate shear modulus (G = rho * c44 = rho * Vs2**2) for both media
    # and their average and difference
    G1 = upper_density_gcm3 * Vs2_up**2
    G2 = lower_density_gcm3 * Vs2_low**2
    G_avg = (G1 + G2) / 2.0
    G_diff = G2 - G1

    # Calculate average and difference for P-wave and S-wave velocities
    a_avg = (Vp0_up + Vp0_low) / 2.0
    a_diff = Vp0_low - Vp0_up
    b_avg = (Vs2_up + Vs2_low) / 2.0

    # Calculate sin²(θ) and tan²(θ)
    sin_sq_theta = np.sin(incident_angles_rad)**2
    tan_sq_theta = np.tan(incident_angles_rad)**2

    # Calculate factor f
    f = (2 * b_avg / a_avg)**2

    # Calculate reflectivity in xz plane (Rxz) — Ruger (1998) Eq. 23
    Rxz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * (G_diff / G_avg - 2 * (gS_low - gS_up)) + d2_low - d2_up) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e2_low - e2_up) * sin_sq_theta * tan_sq_theta)

    # Calculate reflectivity in yz plane (Ryz) — Ruger (1998) Eq. 24
    Ryz = ((Z2 - Z1) / (Z2 + Z1) +
           0.5 * (a_diff / a_avg - f * G_diff / G_avg + d1_low - d1_up) * sin_sq_theta +
           0.5 * (a_diff / a_avg + e1_low - e1_up) * sin_sq_theta * tan_sq_theta)

    return Rxz, Ryz


def zoeppritz_reflectivity(
    cij_upper_layer: np.ndarray,
    cij_lower_layer: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    azimuths_deg: float | np.ndarray,
    polar_deg: float | np.ndarray,
) -> pd.DataFrame:
    """
    Calculates the exact plane-wave reflection and transmission
    coefficients for an incident qP wave at a planar welded interface
    between two arbitrarily anisotropic media (up to triclinic).

    This is the generalisation of the Zoeppritz equations to
    anisotropic media (Fryer & Frazer, 1984; Schoenberg & Protazio,
    1992). For each incidence direction the Christoffel equation is
    solved for the six vertical slownesses compatible with the
    horizontal slowness of the incident wave (Snell's law), the three
    reflected and three transmitted waves are identified from their
    vertical energy flux (or decay direction when evanescent), and the
    welded-contact boundary conditions (continuity of displacement and
    vertical traction) are solved as a 6x6 linear system. Unlike the
    linearised approximation in :func:`reflectivity` (Ruger, 1998),
    this solution is exact for any strength of anisotropy or interface
    contrast, and remains valid beyond the critical angle(s), where
    the coefficients become complex.

    Geometry: the interface is the horizontal plane z=0 with the upper
    medium occupying z>0, so both stiffness tensors must be given in a
    reference frame whose z-axis is the interface normal (rotate them
    beforehand for dipping interfaces). The incident wave propagates
    downwards along (sin(polar)cos(azimuth), sin(polar)sin(azimuth),
    -cos(polar)), i.e. ``polar_deg`` is the incidence angle measured
    from the interface normal and ``azimuths_deg`` the azimuth of the
    incidence plane (0 = xz plane, 90 = yz plane). A grid covering all
    orientations can be generated with
    :func:`pyrockwave.equispaced_S2_grid_offset` (upper hemisphere, degrees),
    discarding points with ``polar_deg == 90``.

    Parameters
    ----------
    cij_upper_layer : numpy.ndarray
        The 6x6 elastic stiffness tensor of the upper (incidence)
        medium in GPa.
    cij_lower_layer : numpy.ndarray
        The 6x6 elastic stiffness tensor of the lower medium in GPa.
    upper_density_gcm3 : float
        Density of the upper medium in g/cm³.
    lower_density_gcm3 : float
        Density of the lower medium in g/cm³.
    azimuths_deg : float or numpy.ndarray
        Azimuth(s) of the incidence plane in degrees, same shape as
        ``polar_deg``.
    polar_deg : float or numpy.ndarray
        Incidence angle(s) in degrees, measured from the interface
        normal. Must be in [0, 90).

    Returns
    -------
    pd.DataFrame
        One row per incidence direction with columns:

        - ``azimuths_deg``, ``polar_deg`` : the input angles.
        - ``Vp_incident_kms`` : qP phase velocity of the incident wave.
        - ``<name>_mag``, ``<name>_phase_deg`` : magnitude and phase
          (degrees) of each displacement-amplitude coefficient, where
          ``<name>`` is Rpp, Rps1, Rps2 (reflected qP, fast qS, slow
          qS) and Tpp, Tps1, Tps2 (transmitted counterparts).

    Notes
    -----
    Coefficients are displacement-amplitude ratios with unit-length
    polarization vectors. Below the critical angle(s) the phase is 0
    or 180 degrees and the conventional signed coefficient is
    recovered as ``mag * cos(radians(phase))``; the full complex value
    is ``mag * exp(1j * radians(phase))``. At normal incidence between
    isotropic media Rpp reduces to the impedance contrast
    (Z2 - Z1) / (Z2 + Z1).

    qP polarizations are oriented along the propagation direction; the
    sign (phase) of the qS polarizations follows a deterministic but
    arbitrary convention, so the phases of the converted-wave
    coefficients are convention-dependent (their magnitudes are not).
    The fast/slow shear labels S1/S2 are assigned by vertical slowness
    and may swap across shear-wave degeneracies (acoustic axes).
    Incidence angles falling exactly on a critical angle can make the
    up/down classification ambiguous, raising a RuntimeError; perturb
    the angle slightly in that case.

    References
    ----------
    Fryer, G.J., & Frazer, L.N. (1984). Seismic waves in stratified
    anisotropic media. Geophysical Journal of the Royal Astronomical
    Society, 78(3), 691-710.
    https://doi.org/10.1111/j.1365-246X.1984.tb05065.x

    Schoenberg, M., & Protazio, J. (1992). 'Zoeppritz' rationalized
    and generalized to anisotropy. Journal of Seismic Exploration,
    1(2), 125-144.
    """

    azimuths_deg, polar_deg = _validate_zoeppritz_reflectivity(
        cij_upper_layer,
        cij_lower_layer,
        upper_density_gcm3,
        lower_density_gcm3,
        azimuths_deg,
        polar_deg,
    )

    # Build downward-pointing unit wavevectors from the incidence angles
    x, y, z = sph2cart(np.deg2rad(azimuths_deg), np.deg2rad(polar_deg))
    wavevectors = np.column_stack((x, y, -z))

    # rearrange Cij → Cijkl; normalise with density
    # Cijkl in GPa and ρ in g/cm^3 gives (km/s)^2
    cijkl_upper = _rearrange_tensor(cij_upper_layer)
    cijkl_lower = _rearrange_tensor(cij_lower_layer)
    chat_upper = cijkl_upper / upper_density_gcm3
    chat_lower = cijkl_lower / lower_density_gcm3

    # Incident qP wave: phase velocity (fastest Christoffel mode) and
    # polarization, oriented along the propagation direction. The
    # slowness vector is s = n / v (s/km).
    eigenvalues, eigenvectors = _calc_eigen(
        _christoffel_matrix(wavevectors, chat_upper)
    )
    vp_incident = np.sqrt(eigenvalues[:, 2])
    pol_incident = eigenvectors[:, 2, :]
    sign = np.sign(np.einsum("ni,ni->n", pol_incident, wavevectors))
    pol_incident = pol_incident * sign[:, np.newaxis]
    slow_incident = wavevectors / vp_incident[:, np.newaxis]

    # Snell's law: the horizontal slowness is shared by all scattered waves
    slow_horizontal = slow_incident.copy()
    slow_horizontal[:, 2] = 0.0

    # Solve for the six vertical slowness branches in each medium and
    # keep the three upgoing (reflected) waves in the upper medium and
    # the three downgoing (transmitted) waves in the lower medium,
    # each sorted as (qP, qS1, qS2).
    q_up, pol_up, slow_up = _vertical_slowness_modes(chat_upper, slow_horizontal)
    q_low, pol_low, slow_low = _vertical_slowness_modes(chat_lower, slow_horizontal)
    upgoing = _upgoing_mask(chat_upper, q_up, pol_up, slow_up)
    downgoing = ~_upgoing_mask(chat_lower, q_low, pol_low, slow_low)
    pol_refl, slow_refl = _select_scattered_modes(
        chat_upper, pol_up, slow_up, upgoing, "reflected"
    )
    pol_trans, slow_trans = _select_scattered_modes(
        chat_lower, pol_low, slow_low, downgoing, "transmitted"
    )

    # Welded-contact boundary conditions: continuity of displacement
    # and vertical traction across z=0 gives a 6x6 linear system per
    # direction for the scattered displacement amplitudes.
    b_incident = _displacement_traction_vectors(
        cijkl_upper, pol_incident[:, np.newaxis, :], slow_incident[:, np.newaxis, :]
    )[:, 0, :]
    b_reflected = _displacement_traction_vectors(cijkl_upper, pol_refl, slow_refl)
    b_transmitted = _displacement_traction_vectors(cijkl_lower, pol_trans, slow_trans)
    system = np.swapaxes(
        np.concatenate((b_reflected, -b_transmitted), axis=1), 1, 2
    )
    coefficients = np.linalg.solve(system, -b_incident[:, :, np.newaxis])[:, :, 0]

    # Assemble DataFrame (magnitude and phase per coefficient)
    df = pd.DataFrame(
        {
            "azimuths_deg": azimuths_deg,
            "polar_deg": polar_deg,
            "Vp_incident_kms": vp_incident,
        }
    )
    for name, coefficient in zip(
        ("Rpp", "Rps1", "Rps2", "Tpp", "Tps1", "Tps2"), coefficients.T
    ):
        df[f"{name}_mag"] = np.abs(coefficient)
        df[f"{name}_phase_deg"] = np.degrees(np.angle(coefficient))

    return df


def schoenberg_muir_layered_medium(
    cij_layer1: np.ndarray,
    cij_layer2: np.ndarray,
    vfrac1: float,
    vfrac2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the effective stiffness and compliance tensors for
    a layered medium using the Schoenberg & Muir approach. This
    function computes the effective stiffness and compliance
    tensors for a medium composed of two alternating thin layers.

    Parameters
    ----------
    cij_layer1 : numpy.ndarray
        6x6 stiffness tensor of the first layer.
    cij_layer2 : numpy.ndarray
        6x6 stiffness tensor of the second layer.
    vfrac1 : float
        Volume (thickness) fraction of the first layer (0 <= vfrac1 <= 1).
    vfrac2 : float
        Volume (thickness) fraction of the second layer (0 <= vfrac2 <= 1).

    Returns
    -------
    effective_stiffness : numpy.ndarray of shape (6, 6)
        Effective stiffness tensor.
    effective_compliance : numpy.ndarray of shape (6, 6)
        Effective compliance tensor.
    effective_stiffness_from_compliance : numpy.ndarray of shape (6, 6)
        Effective stiffness tensor computed by inverting the effective
        compliance. In exact arithmetic this is identical to
        ``effective_stiffness``; it is returned as an independent
        consistency check on the Schoenberg & Muir calculus, and the
        two should agree to within numerical precision.

    References
    ----------
    Schoenberg, M., & Muir, F. (1989). A calculus for finely layered
    anisotropic media. Geophysics, 54(5), 581-589.
    https://doi.org/10.1190/1.1442685

    Nichols, D., Muir, F., & Schoenberg, M. (1989). Elastic
    properties of rocks with multiple sets of fractures. SEG
    Technical Program Expanded Abstracts: 471-474.
    https://doi.org/10.1190/1.1889682

    Notes
    -----
    This is Python reimplementation with improved readability and
    documentation from srb toolbox (see sch_muir_bckus.m file)
    originally written by Kaushik Bandyopadhyay in 2008.
    """

    _validate_schoenberg_muir_layered_medium(cij_layer1, cij_layer2, vfrac1, vfrac2)

    # Extract submatrices (partition) for Cnn, Ctn, Ctt from the stiffness tensor
    # normal
    Cnn1 = cij_layer1[2:5, 2:5]
    Cnn2 = cij_layer2[2:5, 2:5]
    # tangential-normal
    Ctn1 = cij_layer1[[0, 1, 5], 2:5]
    Ctn2 = cij_layer2[[0, 1, 5], 2:5]
    # tangential -> np.ix_(row_indices, col_indices)
    Ctt1 = cij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Ctt2 = cij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]

    # Pre-compute inverses used multiple times
    Cnn1_inv = np.linalg.inv(Cnn1)
    Cnn2_inv = np.linalg.inv(Cnn2)

    # Compute effective normal stiffness
    Cnn = np.linalg.inv(vfrac1 * Cnn1_inv + vfrac2 * Cnn2_inv)

    # Compute effective tangential-normal stiffness
    Ctn = (vfrac1 * Ctn1 @ Cnn1_inv + vfrac2 * Ctn2 @ Cnn2_inv) @ Cnn

    # Compute effective tangential stiffness
    term1 = vfrac1 * Ctt1 + vfrac2 * Ctt2
    term2 = vfrac1 * Ctn1 @ Cnn1_inv @ Ctn1.T + vfrac2 * Ctn2 @ Cnn2_inv @ Ctn2.T
    term3 = (vfrac1 * Ctn1 @ Cnn1_inv + vfrac2 * Ctn2 @ Cnn2_inv) @ Cnn @ (
                vfrac1 * Cnn1_inv @ Ctn1.T + vfrac2 * Cnn2_inv @ Ctn2.T)

    Ctt = term1 - term2 + term3

    # Assemble effective stiffness tensor in full Voigt notation
    effective_stiffness = np.array(
        [
            [Ctt[0, 0], Ctt[0, 1], Ctn[0, 0], Ctn[0, 1], Ctn[0, 2], Ctt[0, 2]],
            [Ctt[1, 0], Ctt[1, 1], Ctn[1, 0], Ctn[1, 1], Ctn[1, 2], Ctt[1, 2]],
            [Ctn[0, 0], Ctn[1, 0], Cnn[0, 0], Cnn[1, 0], Cnn[0, 2], Ctn[2, 0]],
            [Ctn[0, 1], Ctn[1, 1], Cnn[1, 0], Cnn[1, 1], Cnn[1, 2], Ctn[2, 1]],
            [Ctn[0, 2], Ctn[1, 2], Cnn[0, 2], Cnn[1, 2], Cnn[2, 2], Ctn[2, 2]],
            [Ctt[0, 2], Ctt[1, 2], Ctn[2, 0], Ctn[2, 1], Ctn[2, 2], Ctt[2, 2]],
        ]
    )

    # Compute compliances from stiffnesses
    sij_layer1 = np.linalg.inv(cij_layer1)
    sij_layer2 = np.linalg.inv(cij_layer2)

    # Extract submatrices (partition) for Snn, Stn, Stt from the compliance tensor
    # normal
    Snn1 = sij_layer1[2:5, 2:5]
    Snn2 = sij_layer2[2:5, 2:5]
    # tangential-normal
    Stn1 = sij_layer1[[0, 1, 5], 2:5]
    Stn2 = sij_layer2[[0, 1, 5], 2:5]
    # tangential -> np.ix_(row_indices, col_indices)
    Stt1 = sij_layer1[np.ix_([0, 1, 5], [0, 1, 5])]
    Stt2 = sij_layer2[np.ix_([0, 1, 5], [0, 1, 5])]

    # Pre-compute inverses used multiple times
    Stt1_inv = np.linalg.inv(Stt1)
    Stt2_inv = np.linalg.inv(Stt2)

    # Compute effective tangential compliance (needed by Stn and Snn below)
    Stt = np.linalg.inv(vfrac1 * Stt1_inv + vfrac2 * Stt2_inv)

    # Compute effective tangential-normal compliance
    Stn = Stt @ (vfrac1 * Stt1_inv @ Stn1 + vfrac2 * Stt2_inv @ Stn2)

    # Compute effective normal compliance
    Snn = ((vfrac1 * Snn1 + vfrac2 * Snn2)
           - (vfrac1 * Stn1.T @ Stt1_inv @ Stn1 + vfrac2 * Stn2.T @ Stt2_inv @ Stn2)
           + (vfrac1 * Stn1.T @ Stt1_inv + vfrac2 * Stn2.T @ Stt2_inv)
           @ Stt
           @ (vfrac1 * Stt1_inv @ Stn1 + vfrac2 * Stt2_inv @ Stn2))

    # Assemble effective compliance matrix
    effective_compliance = np.array(
        [
            [Stt[0, 0], Stt[0, 1], Stn[0, 0], Stn[0, 1], Stn[0, 2], Stt[0, 2]],
            [Stt[1, 0], Stt[1, 1], Stn[1, 0], Stn[1, 1], Stn[1, 2], Stt[1, 2]],
            [Stn[0, 0], Stn[1, 0], Snn[0, 0], Snn[1, 0], Snn[0, 2], Stn[2, 0]],
            [Stn[0, 1], Stn[1, 1], Snn[1, 0], Snn[1, 1], Snn[1, 2], Stn[2, 1]],
            [Stn[0, 2], Stn[1, 2], Snn[0, 2], Snn[1, 2], Snn[2, 2], Stn[2, 2]],
            [Stt[0, 2], Stt[1, 2], Stn[2, 0], Stn[2, 1], Stn[2, 2], Stt[2, 2]],
        ]
    )

    # Compute stiffness from effective compliance
    effective_stiffness_from_compliance = np.linalg.inv(effective_compliance)

    return effective_stiffness, effective_compliance, effective_stiffness_from_compliance


# =================================================================
# Private helpers for internal use only

def _validate_zoeppritz_reflectivity(
    cij_upper_layer: np.ndarray,
    cij_lower_layer: np.ndarray,
    upper_density_gcm3: float,
    lower_density_gcm3: float,
    azimuths_deg: float | np.ndarray,
    polar_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs for :func:`zoeppritz_reflectivity` and coerce
    the angle inputs to 1D float64 arrays."""
    validate_cij(cij_upper_layer)
    validate_cij(cij_lower_layer)

    if upper_density_gcm3 <= 0 or lower_density_gcm3 <= 0:
        raise ValueError("Densities must be positive.")

    azimuths_deg = np.atleast_1d(np.asarray(azimuths_deg, dtype=np.float64))
    polar_deg = np.atleast_1d(np.asarray(polar_deg, dtype=np.float64))

    if azimuths_deg.ndim != 1 or polar_deg.ndim != 1:
        raise ValueError("azimuths_deg and polar_deg must be 1D arrays (or scalars).")

    if azimuths_deg.shape != polar_deg.shape:
        raise ValueError("azimuths_deg and polar_deg must have the same shape.")

    if np.any(polar_deg < 0) or np.any(polar_deg >= 90):
        raise ValueError("polar_deg (incidence angles) must be in [0, 90).")

    return azimuths_deg, polar_deg


def _vertical_slowness_modes(
    chat: np.ndarray,
    slow_horizontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Christoffel equation for the six vertical slowness
    branches compatible with a fixed horizontal slowness.

    Writing the slowness vector as s = (s1, s2, q) with (s1, s2)
    fixed by Snell's law, the Christoffel equation
    chat_ijkl s_j s_l a_k = a_i becomes a quadratic eigenvalue
    problem in q, (A + q B + q² D) a = 0, with
    A_ik = chat_ijkl s_j s_l - delta_ik (horizontal parts only),
    B_ik = chat_ijk3 s_j + chat_i3kl s_l and D_ik = chat_i3k3.
    It is linearised into a 6x6 companion eigenproblem (equivalent
    to the Fryer & Frazer (1984) system matrix), whose eigenvectors
    stack the polarization a on top of q*a.

    Parameters
    ----------
    chat : numpy.ndarray of shape (3, 3, 3, 3)
        Density-normalised elastic tensor in (km/s)².
    slow_horizontal : numpy.ndarray of shape (n, 3)
        Horizontal slowness vectors (s1, s2, 0) in s/km.

    Returns
    -------
    q : numpy.ndarray of shape (n, 6), complex
        Vertical slownesses (complex for evanescent branches).
    pol : numpy.ndarray of shape (n, 6, 3), complex
        Unit polarization vectors, indexed as (direction, mode, cart).
    slow : numpy.ndarray of shape (n, 6, 3), complex
        Full slowness vectors (s1, s2, q) per mode.
    """
    n = slow_horizontal.shape[0]

    A = (
        np.einsum("nj,ijkl,nl->nik", slow_horizontal, chat, slow_horizontal)
        - np.eye(3)
    )
    B = (
        np.einsum("nj,ijk->nik", slow_horizontal, chat[:, :, :, 2])
        + np.einsum("nl,ikl->nik", slow_horizontal, chat[:, 2, :, :])
    )
    D_inv = np.linalg.inv(chat[:, 2, :, 2])

    companion = np.zeros((n, 6, 6))
    companion[:, :3, 3:] = np.eye(3)
    companion[:, 3:, :3] = -D_inv @ A
    companion[:, 3:, 3:] = -D_inv @ B

    q, eigenvectors = np.linalg.eig(companion)

    # top half of each eigenvector is the (unnormalised) polarization
    pol = np.swapaxes(eigenvectors[:, :3, :], 1, 2)  # (direction, mode, cart)
    pol = pol / np.linalg.norm(pol, axis=2, keepdims=True)

    slow = np.repeat(
        slow_horizontal[:, np.newaxis, :], 6, axis=1
    ).astype(np.complex128)
    slow[:, :, 2] = q

    return q, pol, slow


def _upgoing_mask(
    chat: np.ndarray,
    q: np.ndarray,
    pol: np.ndarray,
    slow: np.ndarray,
) -> np.ndarray:
    """
    Classify the six slowness branches of each direction as upgoing
    (True) or downgoing (False). Propagating waves are classified by
    the sign of the vertical energy flux, <P_z> ∝ Re(chat_3jkl a*_j
    a_k s_l) (the vertical group-velocity component); evanescent waves
    by the sign of Im(q), i.e. by requiring decay away from the
    interface (upgoing waves live in the upper medium, z > 0).
    """
    flux_z = np.real(
        np.einsum("jkl,nmj,nmk,nml->nm", chat[2], np.conj(pol), pol, slow)
    )
    slowness_scale = np.sqrt(np.sum(np.abs(slow) ** 2, axis=2))
    evanescent = np.abs(q.imag) > 1e-6 * slowness_scale

    return np.where(evanescent, q.imag > 0, flux_z > 0)


def _select_scattered_modes(
    chat: np.ndarray,
    pol: np.ndarray,
    slow: np.ndarray,
    keep: np.ndarray,
    wave_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the three slowness branches flagged in ``keep`` and return
    their polarizations and slownesses sorted as (qP, qS1, qS2) with
    the polarization sign/phase conventions applied (see
    :func:`_sort_and_orient_modes`).
    """
    counts = keep.sum(axis=1)
    if not np.all(counts == 3):
        bad = np.flatnonzero(counts != 3)
        raise RuntimeError(
            f"Could not identify exactly three {wave_kind} waves for "
            f"direction indices {bad[:10]}. This can happen exactly at "
            "a critical angle; perturb the incidence angle slightly."
        )

    order = np.argsort(~keep, axis=1, kind="stable")[:, :3]
    pol = np.take_along_axis(pol, order[:, :, np.newaxis], axis=1)
    slow = np.take_along_axis(slow, order[:, :, np.newaxis], axis=1)

    return _sort_and_orient_modes(chat, pol, slow)


def _sort_and_orient_modes(
    chat: np.ndarray,
    pol: np.ndarray,
    slow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort three scattered modes as (qP, qS1, qS2) and fix polarization
    sign/phase conventions.

    The qP mode is the one with the largest longitudinal polarization
    component |a·s|/|s|; the two quasi-shear modes are ordered by
    increasing |vertical slowness| (qS1 = fast). Degenerate shear
    pairs are rebuilt into a flux-decoupled basis (see
    :func:`_decouple_degenerate_shear`). The qP polarization phase is
    rotated so that a·s is real and positive (which reduces to
    'oriented along propagation' for propagating waves); each qS
    polarization is rotated so its largest component is real positive.
    """
    n = pol.shape[0]

    # identify qP: largest longitudinal component
    slowness_norm = np.sqrt(np.sum(np.abs(slow) ** 2, axis=2))
    longitudinality = np.abs(np.einsum("nmk,nmk->nm", pol, slow)) / slowness_norm
    p_idx = np.argmax(longitudinality, axis=1)

    # remaining two modes are quasi-shear; qS1 = smaller |q| (faster)
    mode_idx = np.tile(np.arange(3), (n, 1))
    shear_idx = mode_idx[mode_idx != p_idx[:, np.newaxis]].reshape(n, 2)
    q_shear_abs = np.abs(np.take_along_axis(slow[:, :, 2], shear_idx, axis=1))
    flip = q_shear_abs[:, 0] > q_shear_abs[:, 1]
    shear_idx[flip] = shear_idx[flip, ::-1]

    order = np.column_stack((p_idx, shear_idx))
    pol = np.take_along_axis(pol, order[:, :, np.newaxis], axis=1)
    slow = np.take_along_axis(slow, order[:, :, np.newaxis], axis=1)

    # degenerate shear pairs (any direction in an isotropic medium,
    # acoustic axes in anisotropic media): the eigensolver returns an
    # arbitrary, generally non-orthogonal basis of the 2D shear
    # eigenspace, whose members would exchange energy. Rebuild a
    # flux-decoupled pair so the two shear coefficients are physically
    # meaningful and their energy fluxes additive.
    q_gap = np.abs(slow[:, 1, 2] - slow[:, 2, 2])
    degenerate = q_gap < 1e-6 * slowness_norm[:, 1]
    if np.any(degenerate):
        pol[degenerate] = _decouple_degenerate_shear(
            chat, pol[degenerate], slow[degenerate]
        )

    # qP phase convention: a·s real positive (along propagation)
    zeta = np.einsum("nk,nk->n", pol[:, 0, :], slow[:, 0, :])
    zeta = np.where(np.abs(zeta) == 0, 1.0, zeta)
    pol[:, 0, :] = pol[:, 0, :] * (np.conj(zeta) / np.abs(zeta))[:, np.newaxis]

    # qS phase convention: largest component real positive
    for mode in (1, 2):
        largest_idx = np.argmax(np.abs(pol[:, mode, :]), axis=1)
        largest = np.take_along_axis(
            pol[:, mode, :], largest_idx[:, np.newaxis], axis=1
        )[:, 0]
        pol[:, mode, :] = (
            pol[:, mode, :] * (np.conj(largest) / np.abs(largest))[:, np.newaxis]
        )

    return pol, slow


def _decouple_degenerate_shear(
    chat: np.ndarray,
    pol: np.ndarray,
    slow: np.ndarray,
) -> np.ndarray:
    """
    Rebuild the two quasi-shear polarizations of shear-degenerate
    directions into an orthonormal, flux-decoupled pair.

    Within a degenerate eigenspace any linear combination of the two
    shear modes is a valid plane wave, so the pair returned by the
    eigensolver is arbitrary and generally neither orthogonal nor
    energetically independent. The pair is first orthonormalised
    (Gram-Schmidt) and then rotated to diagonalise the 2x2 Hermitian
    vertical energy-flux matrix H_mn = Re-part of a*_m,i chat_i3kl
    a_n,k s_l, which decouples the vertical energy transport of the
    two modes.

    Parameters are the density-normalised tensor ``chat`` and the
    (already sorted) polarizations (n, 3, 3) and slownesses (n, 3, 3)
    of the shear-degenerate subset; modes 1 and 2 are the shear pair.
    Returns the updated polarization array.
    """
    a1 = pol[:, 1, :]
    a2 = pol[:, 2, :]

    # orthonormalise the pair (Hermitian Gram-Schmidt)
    overlap = np.einsum("nk,nk->n", np.conj(a1), a2)
    a2 = a2 - overlap[:, np.newaxis] * a1
    a2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    pair = np.stack((a1, a2), axis=1)  # (n, 2, 3)

    # 2x2 Hermitian vertical flux matrix within the eigenspace
    traction = np.einsum(
        "ikl,nmk,nml->nmi", chat[:, 2, :, :], pair, slow[:, 1:3, :]
    )
    M = np.einsum("nmi,noi->nmo", np.conj(pair), traction)
    H = 0.5 * (M + np.conj(np.swapaxes(M, 1, 2)))

    # rotate the pair by the unitary eigenbasis of H
    _, U = np.linalg.eigh(H)
    pol[:, 1:3, :] = np.einsum("nkm,nki->nmi", U, pair)

    return pol


def _displacement_traction_vectors(
    cijkl: np.ndarray,
    pol: np.ndarray,
    slow: np.ndarray,
) -> np.ndarray:
    """
    Build the 6-component displacement-traction vectors b = [a, t]
    with t_i = c_i3kl a_k s_l, the traction exerted on horizontal
    planes (the common factor iω is dropped as it cancels in the
    boundary-condition system). ``pol`` and ``slow`` have shape
    (n, m, 3); the result has shape (n, m, 6).
    """
    traction = np.einsum("ikl,nmk,nml->nmi", cijkl[:, 2, :, :], pol, slow)

    return np.concatenate((pol, traction), axis=2)


def _validate_schoenberg_muir_layered_medium(
    cij_layer1: np.ndarray,
    cij_layer2: np.ndarray,
    vfrac1: float,
    vfrac2: float,
) -> None:
    """Validate inputs for :func:`schoenberg_muir_layered_medium`."""
    if not isinstance(cij_layer1, np.ndarray) or not isinstance(cij_layer2, np.ndarray):
        raise TypeError("cij_layer1 and cij_layer2 must be numpy arrays.")

    if cij_layer1.shape != (6, 6) or cij_layer2.shape != (6, 6):
        raise ValueError("cij_layer1 and cij_layer2 must be 6x6 arrays.")

    if not np.allclose(cij_layer1, cij_layer1.T):
        raise ValueError("the elastic tensor 1 is not symmetric!")

    if not np.allclose(cij_layer2, cij_layer2.T):
        raise ValueError("the elastic tensor 2 is not symmetric!")

    if not isinstance(vfrac1, (int, float)) or not isinstance(vfrac2, (int, float)):
        raise TypeError("vfrac1 and vfrac2 must be numbers.")

    if not (0 <= vfrac1 <= 1) or not (0 <= vfrac2 <= 1):
        raise ValueError("Volume fractions must be between 0 and 1.")

    if not np.isclose(vfrac1 + vfrac2, 1, rtol=1e-03):
        raise ValueError("Volume fractions must sum (approximately) to 1.")


# End of file
