# -*- coding: utf-8 -*-
#######################################################################
# This file is part of PyRockWave Python module                       #
#                                                                     #
# Filename: layered_media.py                                          #
# Description: TODO                                                   #
#                                                                     #
# Copyright (c) 2023                                                  #
#                                                                     #
# PyRockWave is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published   #
# by the Free Software Foundation, either version 3 of the License,   #
# or (at your option) any later version.                              #
#                                                                     #
# PyRockWave is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of      #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
# GNU General Public License for more details.                        #
#                                                                     #
# You should have received a copy of the GNU General Public License   #
# along with PyRockWave. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                     #
# Author:                                                             #
# Marco A. Lopez-Sanchez, http://orcid.org/0000-0002-0261-9267        #
# Email: lopezmarco [to be found at] uniovi.es                        #
# Website: https://marcoalopez.github.io/PyRockWave/                  #
# Project Repository: https://github.com/marcoalopez/PyRockWave       #
#######################################################################

# Import statements
import numpy as np


# Function definitions
def snell(vp1, vp2, vs1, vs2, theta1):
    """
    Calculates the angles of and refraction and reflection for an incident
    P-wave in a two-layered system.

    Parameters
    ----------
    vp1 : float or array-like
        Compressional velocity of upper layer.
    vp2 : float or array-like
        Compressional velocity of lower layer.
    vs1 : float or array-like
        Shear velocity of upper layer.
    vs2 : float or array-like
        Shear velocity of lower layer.
    theta1 : float or array-like
        Angle of incidence of P-wave in upper layer in degrees.

    Returns
    -------
    theta2 : float or array-like
        Angle of refraction for P-wave in lower layer in degrees.
    thetas1 : float or array-like
        Angle of reflection for S-wave in upper layer in degrees.
    thetas2 : float or array-like
        Angle of refraction for S-wave in lower layer in degrees.
    p : float or array-like
        Ray parameter.

    """
    
    # Convert angles to radians
    theta1 = np.radians(theta1)
    
    # Calculate ray parameter using Snell's law
    p = np.sin(theta1) / vp1
    
    # Calculate reflection and refraction angles using Snell's law
    thetas1 = np.arcsin(p * vs1)
    theta2 = np.arcsin(p * vp2)
    thetas2 = np.arcsin(p * vs2)

    return theta2, thetas1, thetas2, p


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    """
    The Zoeppritz equations describe seismic wave energy partitioning
    at an interface, for example the boundary between two different
    rocks. The equations relate the amplitude of incident P-waves to
    reflected and refracted P- and S-waves at a plane interface for
    a given angle of incidence.

    Parameters
    ----------
    vp1 : float or array-like
        Compressional velocity of upper layer.
    vs1 : float or array-like
        Shear velocity of upper layer.
    rho1 : float or array-like
        Density of upper layer.
    vp2 : float or array-like
        Compressional velocity of lower layer.
    vs2 : float or array-like
        Shear velocity of lower layer.
    rho2 : float or array-like
        Density of lower layer.
    theta1 : float or array-like
        Angle of incidence for P wave in upper layer in degrees.

    Returns
    -------
    Rpp : float or array-like
        Amplitude coefficients for reflected P-waves.

    """
    
    # Convert angles to radians for numpy functions
    theta1 = np.radians(theta1)
    
    # Calculate reflection and refraction angles using Snell's law
    theta2, thetas1, thetas2, _ = snell(vp1, vp2, vs1, vs2, theta1)

    # Define matrices M and N based on Zoeppritz equations
    M = np.array([
        [-np.sin(theta1), -np.cos(thetas1), np.sin(theta2), np.cos(thetas2)],
        [np.cos(theta1), -np.sin(thetas1), np.cos(theta2), -np.sin(thetas2)],
        [2*rho1*vs1*np.sin(thetas1)*np.cos(theta1), rho2*vs1*(1-2*np.sin(thetas2)**2),
         2*rho2*vs2*np.sin(thetas2)*np.cos(theta2), rho2*vs2*(1-2*np.sin(thetas2)**2)],
        [-rho1*vp1*(1-2*np.sin(thetas1)**2), rho1*vs1*np.sin(2*thetas1),
         rho2*vp2*(1-2*np.sin(thetas2)**2), -rho2*vs2*np.sin(2*thetas2)]
    ])

    N = np.array([
        [np.sin(theta1), np.cos(thetas1), -np.sin(theta2), -np.cos(thetas2)],
        [np.cos(theta1), -np.sin(thetas1), np.cos(theta2), -np.sin(thetas2)],
        [  0.5*rho1*vs1*np.sin(  0.5*thetas1)*np.cos(  0.5*theta1),
           0.5*rho2*vs1*(0.5-0.5*np.sin(  0.5*thetas2)**0.5),
           0.5*rho2*vs2*np.sin(  0.5*thetas2)*np.cos(  0.5*theta2),
           0.5*rho2*vs2*(0.5-0.5*np.sin(  0.5*thetas3)**0.5)],
        [rho3*vp3*(0.5-0.5*np.sin(  0.5*thetas3)**0.5),
         -rho3*vs3*np.sin(  0.5*thetas3),
         -rho4*vp4*(0.5-0.5*np.sin(  0.5*thetas4)**0.5),
          rho4*vs4*np.sin(  0.5*thetas4)]
    ])

    # Solve system of equations to find amplitude coefficients
    Z = np.linalg.solve(M, N)

    return Z[0]


# End of file
