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
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Function definitions
def visible_mask(
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


def culled_quiver(
    ax: Axes3D,
    points: np.ndarray,
    vectors: np.ndarray,
    margin: float = 0.05,
    interactive: bool = True,
    **quiver_kwargs,
) -> Line3DCollection:
    """
    Draw a 3D quiver showing only the arrows anchored on the
    camera-facing hemisphere (see visible_mask).

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
        Visibility threshold passed to visible_mask, by default 0.05.
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
        mask = visible_mask(points, ax, margin=margin)
        state["artist"] = ax.quiver(
            *points[mask].T, *vectors[mask].T, **quiver_kwargs
        )
        ax.figure.canvas.draw_idle()

    redraw()

    if interactive:
        ax.figure.canvas.mpl_connect("button_release_event", redraw)

    return state["artist"]


# End of file
