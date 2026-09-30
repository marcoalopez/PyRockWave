# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# Filename: plots.py                                                          #
# Description: This module contains a few custom plots for the PyRockWave     #
# module.                                                                     #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2026-present, Marco A. Lopez-Sanchez. All rights reserved.    #
#                                                                             #
# PyRockWave is free software: you can redistribute it and/or modify          #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                        #
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
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pyrockwave.utils.coordinates import equispaced_S2_grid


# Function definitions
def velocities_isometric(
    wavevectors: np.ndarray,
    velocities: pd.DataFrame,
    polarizations: np.ndarray
) -> tuple[plt.Figure, Any]:
    """
    Plot Vp, Vs1 and Vs2 with polarizations using an isometric view,
    a 2D representation of a 3D object where the x, y, and z axes are
    equally spaced at 120° angles.

    Parameters
    ----------
    wavevectors : np.ndarray
        _description_
    velocities : pd.DataFrame
        _description_
    polarizations : np.ndarray
        _description_

    Returns
    -------
    tuple[plt.Figure, Any]
        _description_
    """

    fig, axes = plt.subplots(
        figsize=(6.4 * 3, 7.0),
        ncols=3,
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    (ax1, ax2, ax3) = axes

    # ===========================================================================
    # Vp (axe 1)
    # ===========================================================================
    # Set the projection and view
    ax1.set_proj_type("ortho")
    ax1.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=45)

    Vp = ax1.scatter(
        wavevectors[:, 0],
        wavevectors[:, 1],
        wavevectors[:, 2],
        c=velocities["Vp_phase_kms"],
        cmap="Spectral_r",
    )

    # Show reference frame ==================================================
    # add small x, y, z direction arrows originating from the sphere's surface
    axis_length = 0.75
    origins = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = origins  # arrows point radially outward from the surface

    ax1.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=axis_length,
        color="black",
        linewidth=1.5,
        arrow_length_ratio=0.15,
        zorder=10,
    )

    label_offset = 1 + axis_length * 1.2
    for axis_label, (lx, ly, lz) in zip("xyz", origins * label_offset):
        ax1.text(
            lx,
            ly,
            lz,
            axis_label,
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )

    # add color bar
    cbar1 = fig.colorbar(Vp, ax=ax1, shrink=0.6, location="bottom")
    cbar1.set_label("Vp velocity (km/s)", fontsize=22)
    cbar1.ax.tick_params(labelsize=22)

    # show relevant info
    Vp_min = velocities["Vp_phase_kms"].min()
    Vp_max = velocities["Vp_phase_kms"].max()
    Vp_anis = 200 * (Vp_max - Vp_min) / (Vp_max + Vp_min)
    ax1.text2D(x=0, y=-0.07, s=f"min = {Vp_min:.2f} km/s", fontsize=20, ha="center")
    ax1.text2D(x=0, y=-0.08, s=f"max = {Vp_max:.2f} km/s", fontsize=20, ha="center")
    ax1.text2D(x=0, y=-0.09, s=f"anisotropy = {Vp_anis:.1f} %", fontsize=20, ha="center")

    # tweak plot
    ax1.axis("off")
    ax1.set_box_aspect([1, 1, 1])  # enforce equal aspect ratio


    # ===========================================================================
    # Vs1 - fast (axe 2)
    # ===========================================================================
    # Set the projection and view
    ax2.set_proj_type("ortho")
    ax2.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=45)

    Vs1 = ax2.scatter(
        wavevectors[:, 0],
        wavevectors[:, 1],
        wavevectors[:, 2],
        c=velocities["Vs1_phase_kms"],
        cmap="Spectral_r",
    )

    # S-wave polarizations, camera-facing hemisphere only
    coarse_grid = equispaced_S2_grid(ang_spacing_deg=10, include_axes=True)
    subset = np.unique(cKDTree(wavevectors).query(coarse_grid)[1])
    _culled_quiver(
        ax2,
        wavevectors[subset],
        polarizations[subset, 1],
        color="black",
        pivot="middle",
        length=0.15,
        arrow_length_ratio=0,
        label="S1-wave",
        alpha=0.5,
    )

    # Show reference frame ==================================================
    # add small x, y, z direction arrows originating from the sphere's surface
    ax2.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=axis_length,
        color="black",
        linewidth=1.5,
        arrow_length_ratio=0.15,
        zorder=10,
    )

    for axis_label, (lx, ly, lz) in zip("xyz", origins * label_offset):
        ax2.text(
            lx,
            ly,
            lz,
            axis_label,
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )

    # add color bar
    cbar2 = fig.colorbar(Vs1, ax=ax2, shrink=0.6, location="bottom")
    cbar2.set_label("Vs1 velocity (km/s)", fontsize=22)
    cbar2.ax.tick_params(labelsize=22)

    # show relevant info
    Vs1_min = velocities["Vs1_phase_kms"].min()
    Vs1_max = velocities["Vs1_phase_kms"].max()
    Vs1_anis = 200 * (Vs1_max - Vs1_min) / (Vs1_max + Vs1_min)
    ax2.text2D(x=0, y=-0.07, s=f"min = {Vs1_min:.2f} km/s", fontsize=20, ha="center")
    ax2.text2D(x=0, y=-0.08, s=f"max = {Vs1_max:.2f} km/s", fontsize=20, ha="center")
    ax2.text2D(x=0, y=-0.09, s=f"anisotropy = {Vs1_anis:.1f} %", fontsize=20, ha="center")

    # tweak plot
    ax2.axis("off")
    ax2.set_box_aspect([1, 1, 1])  # enforce equal aspect ratio


    # ===========================================================================
    # Vs2 - slow (axe 3)
    # ===========================================================================
    # Set the projection and view
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=45)

    Vs2 = ax3.scatter(
        wavevectors[:, 0],
        wavevectors[:, 1],
        wavevectors[:, 2],
        c=velocities["Vs2_phase_kms"],
        cmap="Spectral_r",
    )

    # S-wave polarizations, camera-facing hemisphere only
    _culled_quiver(
        ax3,
        wavevectors[subset],
        polarizations[subset, 1],
        color="black",
        pivot="middle",
        length=0.15,
        arrow_length_ratio=0,
        label="S1-wave",
        alpha=0.5,
    )

    # Show reference frame ==================================================
    # add small x, y, z direction arrows originating from the sphere's surface
    ax3.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=axis_length,
        color="black",
        linewidth=1.5,
        arrow_length_ratio=0.15,
        zorder=10,
    )

    for axis_label, (lx, ly, lz) in zip("xyz", origins * label_offset):
        ax3.text(
            lx,
            ly,
            lz,
            axis_label,
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )

    # add color bar
    cbar3 = fig.colorbar(Vs2, ax=ax3, shrink=0.6, location="bottom")
    cbar3.set_label("Vs2 velocity (km/s)", fontsize=22)
    cbar3.ax.tick_params(labelsize=22)

    # show relevant info
    Vs2_min = velocities["Vs2_phase_kms"].min()
    Vs2_max = velocities["Vs2_phase_kms"].max()
    Vs2_anis = 200 * (Vs2_max - Vs2_min) / (Vs2_max + Vs2_min)
    ax3.text2D(x=0, y=-0.07, s=f"min = {Vs2_min:.2f} km/s", fontsize=20, ha="center")
    ax3.text2D(x=0, y=-0.08, s=f"max = {Vs2_max:.2f} km/s", fontsize=20, ha="center")
    ax3.text2D(x=0, y=-0.09, s=f"anisotropy = {Vs2_anis:.1f} %", fontsize=20, ha="center")

    # tweak plot
    ax3.axis("off")
    ax3.set_box_aspect([1, 1, 1])  # enforce equal aspect ratio

    return fig, axes


def full_isometric(
    wavevectors: np.ndarray,
    velocities: pd.DataFrame,
    polarizations: np.ndarray
) -> tuple[plt.Figure, Any]:
    """
    Plot Vp, Vs1 and Vs2 (with polarizations), Vp/Vs ratios and shear
    wave splitting (SWS) using an isometric view, a 2D representation
    of a 3D object where the x, y, and z axes are equally spaced at
    120° angles. Layout:

                     Vp   |  Vs1   | Vs2
                   Vp/Vs1 | Vp/Vs2 | SWS

    Parameters
    ----------
    wavevectors : np.ndarray
        _description_
    velocities : pd.DataFrame
        _description_
    polarizations : np.ndarray
        _description_

    Returns
    -------
    tuple[plt.Figure, Any]
        _description_
    """

    fig, axes = plt.subplots(
        figsize=(6.4 * 3, 7.0 * 2),
        ncols=3,
        nrows=2,
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    # TODO

    return fig, axes


def velocities_stereo(
    wavevectors: np.ndarray,
    velocities: pd.DataFrame,
    polarizations: np.ndarray,
    hemisphere: str = "upper",
    north: str = "X",
    east: str = "Y",
) -> tuple[plt.Figure, Any]:
    """


    Parameters
    ----------
    wavevectors : np.ndarray
        _description_
    velocities : pd.DataFrame
        _description_
    polarizations : np.ndarray
        _description_
    hemisphere : str, optional
        _description_, by default "upper"
    north : str, optional
        _description_, by default "X"
    east : str, optional
        _description_, by default "Y"

    Returns
    -------
    tuple[plt.Figure, Any]
        _description_
    """

    pass


def full_stereo(
    wavevectors: np.ndarray,
    velocities: pd.DataFrame,
    polarizations: np.ndarray,
    hemisphere: str = "upper",
    north: str = "X",
    east: str = "Y",
) -> tuple[plt.Figure, Any]:
    """


    Parameters
    ----------
    wavevectors : np.ndarray
        _description_
    velocities : pd.DataFrame
        _description_
    polarizations : np.ndarray
        _description_
    hemisphere : str, optional
        _description_, by default "upper"
    north : str, optional
        _description_, by default "X"
    east : str, optional
        _description_, by default "Y"

    Returns
    -------
    tuple[plt.Figure, Any]
        _description_
    """

    pass


# =================================================================
# Private helpers for internal use only

def _visible_mask(
    points: np.ndarray,
    ax: Axes3D,
    margin: float = 0.05,
) -> np.ndarray:
    """
    Return a boolean mask of the unit vectors that lie on the
    camera-facing hemisphere of a 3D axes.

    A point on the unit sphere faces the camera when its dot product
    with the viewing direction (derived from the axes' current azimuth
    and elevation) is positive. Use it to hide markers or arrows on
    'the back of the sphere', which matplotlib would otherwise draw:
    mplot3d has no depth buffer, so an opaque surface does not occlude
    other artists.

    Parameters
    ----------
    points : numpy.ndarray of shape (n, 3)
        Unit vectors (points on the unit sphere).
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes whose current view defines visibility. Set the
        view (ax.view_init) before calling this function. The test is
        exact for orthographic projection (ax.set_proj_type("ortho"));
        with the default perspective projection, keep a small margin.
    margin : float, optional
        Visibility threshold on the dot product, by default 0.05.
        Values of ~0.05-0.1 also trim arrows sitting on the sphere's
        silhouette, which otherwise look ragged.

    Returns
    -------
    numpy.ndarray of shape (n,), dtype bool
        True where the point faces the camera.

    Raises
    ------
    ValueError
        If points is not a numpy array of shape (n, 3).
    """

    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a numpy array of shape (n, 3).")

    azim, elev = np.deg2rad(ax.azim), np.deg2rad(ax.elev)
    view_direction = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ]
    )

    return points @ view_direction > margin


def _culled_quiver(
    ax: Axes3D,
    points: np.ndarray,
    vectors: np.ndarray,
    margin: float = 0.05,
    interactive: bool = True,
    **quiver_kwargs,
) -> Line3DCollection:
    """
    Draw a 3D quiver showing only the arrows anchored on the
    camera-facing hemisphere (see _visible_mask).

    Set the view (ax.view_init) before calling this function: the
    arrows are culled for the axes' current camera position. When
    ``interactive`` is True and an interactive backend is in use, the
    quiver is automatically re-culled after each mouse rotation
    (redraw on button release); with static backends (e.g. inline
    notebook figures) the callback is simply never triggered.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes to draw on.
    points : numpy.ndarray of shape (n, 3)
        Arrow anchor points (unit vectors on the sphere).
    vectors : numpy.ndarray of shape (n, 3)
        Arrow directions, same shape as points.
    margin : float, optional
        Visibility threshold passed to _visible_mask, by default 0.05.
    interactive : bool, optional
        Whether to re-cull the arrows after interactive rotations,
        by default True.
    **quiver_kwargs
        Keyword arguments forwarded to ax.quiver (color, pivot,
        length, alpha, label, ...).

    Returns
    -------
    mpl_toolkits.mplot3d.art3d.Line3DCollection
        The quiver artist currently drawn (replaced on interactive
        redraws).

    Raises
    ------
    ValueError
        If points is not a numpy array of shape (n, 3), or vectors
        does not have the same shape as points.
    """

    if not isinstance(vectors, np.ndarray) or vectors.shape != points.shape:
        raise ValueError("vectors must be a numpy array with the same shape as points.")

    state = {"artist": None}

    def redraw(event=None) -> None:
        # ignore mouse events released over other axes of the figure
        if event is not None and event.inaxes is not ax:
            return
        if state["artist"] is not None:
            state["artist"].remove()
        mask = _visible_mask(points, ax, margin=margin)
        state["artist"] = ax.quiver(
            *points[mask].T, *vectors[mask].T, **quiver_kwargs
        )
        ax.figure.canvas.draw_idle()

    redraw()

    if interactive:
        ax.figure.canvas.mpl_connect("button_release_event", redraw)

    return state["artist"]


# End of file
