# =========================================================================== #
# PyRockWave: A Python Module for modelling elastic properties                #
# of Earth materials.                                                         #
#                                                                             #
# SPDX-License-Identifier: GPL-3.0-or-later                                   #
# Copyright (c) 2026, Marco A. Lopez-Sanchez. All rights reserved.            #
#                                                                             #
# Author: Marco A. Lopez-Sanchez                                              #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Email: lopezmarco [to be found at] uniovi dot es                            #
# Website: https://marcoalopez.github.io/PyRockWave/                          #
# Repository: https://github.com/marcoalopez/PyRockWave                       #
# =========================================================================== #
"""
PyRockWave: A Python Module for modelling elastic properties of Earth materials.
"""

__version__ = "0.1.0"

# Re-export the most commonly used functions at the package level.
from .elastic_tensor import ElasticProps
from .decomposition import decompose_Cij, calc_percentages
from .christoffel import phase_seismic_properties, full_seismic_properties
from .layered_media import (
    calc_reflectivity,
    schoenberg_muir_layered_medium,
    zoeppritz_reflectivity,
)
from .anisotropic_models import (
    weak_polar_anisotropy,
    polar_anisotropy,
    orthotropic_azimuthal_anisotropy,
)
from .utils.coordinates import (
    sph2cart,
    cart2sph,
    equispaced_S2_grid,
    equispaced_S2_grid_offset,
)
from .utils.tensor_tools import rotate_stiffness_tensor


__all__ = [
    "ElasticProps",
    "decompose_Cij",
    "calc_percentages",
    "phase_seismic_properties",
    "full_seismic_properties",
    "calc_reflectivity",
    "schoenberg_muir_layered_medium",
    "zoeppritz_reflectivity",
    "weak_polar_anisotropy",
    "polar_anisotropy",
    "orthotropic_azimuthal_anisotropy",
    "sph2cart",
    "cart2sph",
    "equispaced_S2_grid",
    "equispaced_S2_grid_offset",
    "rotate_stiffness_tensor",
    "__version__",
]
