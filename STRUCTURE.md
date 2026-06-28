# PyRockWave — Repository Structure

> Scope: package layout, module/function inventory, and internal dependencies
> after the reorganisation into the `pyrockwave` package. Function *correctness*
> is out of scope for this document.

## Package layout

```
PyRockWave/
├── src/
│   ├── pyrockwave/                 # the installable package
│   │   ├── __init__.py             # version + public API re-exports
│   │   ├── elastic_tensor.py       # ElasticProps dataclass
│   │   ├── decomposition.py        # Browaeys & Chevrot symmetry-class tensor decomposition
│   │   ├── anisotropic_models.py   # analytic anisotropy models (Thomsen, Tsvankin, ...)
│   │   ├── averaging_schemes.py    # Voigt/Reuss volume- and CPO-weighted averages
│   │   ├── christoffel.py          # Christoffel equation: phase/group seismic properties
│   │   ├── layered_media.py        # reflectivity & layered-medium (Schoenberg–Muir)
│   │   ├── ultrasonic.py           # ultrasonic signal processing
│   │   ├── plotting/               # subpackage — custom plots
│   │   │   ├── __init__.py         # (empty)
│   │   │   └── plots.py            # STUB: no functions yet
│   │   └── utils/                  # subpackage — generic helpers
│   │       ├── __init__.py         # (empty)
│   │       ├── coordinates.py      # spherical/cartesian conversions, S2 grids
│   │       ├── tensor_tools.py     # tensor rearrange/Voigt/rotation helpers
│   │       └── validation.py       # input validators (validate_cij, validate_wavevectors)
│   └── deprecated/                 # NOT part of the package; git-ignored, excluded from build
│       ├── christoffel_old.py
│       └── decompose.py
├── tests/
│   └── test_reflectivity_vs_graebner.py
├── notebooks/                      # examples + exploratory/dev notebooks
├── img/
├── pyproject.toml                  # build metadata (setuptools, src-layout)
├── pixi.toml                       # pixi dev environment
├── CITATION.cff
├── README.md
└── LICENSE (GPL-3.0, code) / License.CC-BY4.txt (docs)
```

Layout style: **src-layout** with a mix of flat submodules (top of `pyrockwave/`)
and feature-based subpackages (`plotting/`, `utils/`).

## Public API (re-exported by `pyrockwave/__init__.py`)

| Symbol | Source module |
|---|---|
| `ElasticProps` | `elastic_tensor` |
| `decompose_Cij`, `calc_percentages` | `decomposition` |
| `phase_seismic_properties`, `full_seismic_properties` | `christoffel` |
| `calc_reflectivity`, `schoenberg_muir_layered_medium` | `layered_media` |
| `weak_polar_anisotropy`, `polar_anisotropy`, `orthotropic_azimuthal_anisotropy` | `anisotropic_models` |
| `sph2cart`, `cart2sph`, `equispaced_S2_grid` | `utils.coordinates` |
| `rotate_stiffness_tensor` | `utils.tensor_tools` |
| `__version__` | `pyrockwave` (`0.1.0`) |

Not re-exported (importable via their full path): `averaging_schemes`,
`ultrasonic`, `utils.validation`, `plotting.plots`.

## Module / function inventory

Public functions/classes only; names with a leading underscore are private helpers.

### `elastic_tensor`
- `ElasticProps` (class) — encapsulates the elastic properties of a crystalline material at a given pressure/temperature.

### `decomposition`
- `decompose_Cij` — decomposes an elastic tensor into its symmetry-class components (Browaeys & Chevrot formulation).
- `calc_percentages` — computes each symmetry class's percentage contribution to the decomposition.
- `_tensor_to_vector` *(private)* — converts a 6×6 Voigt tensor to the 21-component elastic vector.
- `_vector_to_tensor` *(private)* — inverse of the above: rebuilds a 6×6 tensor from a 21-component vector.
- `_orthogonal_projector` *(private)* — builds the 21-D projection matrix that isolates a given symmetry class.

### `anisotropic_models`
- `weak_polar_anisotropy` — body-wave velocities vs. direction under weak (Thomsen) polar anisotropy.
- `polar_anisotropy` — body-wave velocities vs. direction for general (non-weak) polar anisotropy.
- `orthotropic_azimuthal_anisotropy` — P-wave velocity vs. direction for azimuthal orthotropic anisotropy.
- `Thomsen_params` — estimates the Thomsen parameters from a stiffness tensor.
- `tsvankin_params` — estimates the Tsvankin (weak orthorhombic) parameters from a stiffness tensor. **Canonical implementation** (re-used by `layered_media`).
- `HaoStovas_params` — estimates the modified Hao & Stovas (2016) parameters.

### `averaging_schemes`
- `voigt_volume_weighted_average` — Voigt (stiffness) average over volume fractions of minerals.
- `reuss_volume_weighted_average` — Reuss (compliance) average over volume fractions of minerals.
- `voigt_CPO_weighted_average` — Voigt average of a mineral tensor weighted by a crystallographic orientation distribution.
- `reuss_CPO_weighted_average` — Reuss counterpart of the CPO-weighted average.

### `christoffel`
- `phase_seismic_properties` — phase velocities and shear-wave splitting for an array of directions.
- `full_seismic_properties` — phase + group velocities, enhancement factors, and power-flow angles in one call.
- `calc_phase_velocities` — phase velocities of a monochromatic wave from Christoffel eigenvalues.
- `calc_spherical_angles` — converts direction vectors to spherical angles (degrees).
- `calc_group_velocities` — group-velocity vectors, magnitudes, directions, and power-flow angles.
- `calc_enhancement_factor` — exact analytical enhancement factor (Jaeken & Cottenier 2016).
- `calc_power_flow_angles` — angle (degrees) between phase (group) propagation directions.

### `layered_media`
- `snell` — refraction/reflection angles for an incident wave across an interface.
- `calc_reflectivity` — convenience wrapper computing PP reflectivity in both symmetry planes.
- `reflectivity` — symmetry-plane P-wave reflectivity for anisotropic interfaces (Rüger).
- `tsvankin_params` — re-exported from `anisotropic_models` (single canonical implementation); available here for backward compatibility.
- `schoenberg_muir_layered_medium` — effective stiffness/compliance of a finely layered medium (Schoenberg–Muir).

### `ultrasonic`
- `process_signal` — pre-processes a pulse-echo ultrasound signal (crop, detrend, filter).
- `estimate_bandpass` — estimates band-pass corner frequencies from a signal's spectrum.
- `estimate_bandpass_centroid` — estimates the band-pass band from the spectral centroid.
- `trigger_sta_lta` — computes the STA/LTA ratio for arrival-time picking.

### `utils.coordinates`
- `sph2cart` — spherical/polar (magnitude, azimuth, polar) → Cartesian coordinates.
- `cart2sph` — Cartesian → spherical coordinates.
- `equispaced_S2_grid` — approximately equispaced grid of directions on the unit sphere.
- `equispaced_S2_grid_fsa` — Fibonacci-spiral variant of the equispaced S² grid.

### `utils.tensor_tools`
- `rotate_stiffness_tensor` — rotates a stiffness matrix (Voigt) or rank-4 tensor by a given rotation.
- `_rearrange_tensor` *(private, used cross-module)* — expands a 6×6 Voigt matrix to the 3×3×3×3 rank-4 tensor.
- `_tensor_in_voigt` *(private, used cross-module)* — collapses a 3×3×3×3 tensor back to 6×6 Voigt form.

### `utils.validation`
- `validate_cij` — validates a 6×6 Voigt stiffness matrix (shape/symmetry).
- `validate_wavevectors` — validates a wavevector array's shape/contents.

### `plotting.plots`
- *(stub — no functions defined yet)*

## Internal dependency graph

Arrows show "imports from". External deps (numpy, scipy, pandas, matplotlib)
are omitted.

```
__init__            → elastic_tensor, decomposition, christoffel, layered_media,
                      anisotropic_models, utils.coordinates, utils.tensor_tools

christoffel         → utils.tensor_tools (_rearrange_tensor)
                      utils.coordinates  (sph2cart)
                      utils.validation   (validate_cij)

averaging_schemes   → utils.tensor_tools (_rearrange_tensor, _tensor_in_voigt)

layered_media       → anisotropic_models (tsvankin_params)

elastic_tensor      → decomposition (decompose_Cij, calc_percentages)
decomposition       → (no internal deps)
anisotropic_models  → (no internal deps)
ultrasonic          → (no internal deps)
utils.coordinates   → (no internal deps)
utils.tensor_tools  → (no internal deps)
utils.validation    → (no internal deps)
plotting.plots      → (no internal deps)
```

Dependency leaves (depend on nothing internal): `utils.coordinates`,
`utils.tensor_tools`, `utils.validation`, `anisotropic_models`, `decomposition`,
`ultrasonic`. No import cycles. `utils.*` is the shared foundation; `christoffel`
is the most connected module.

## Known structural issues (see review)

- `plotting/plots.py` is still an empty stub (the `plotting/` subpackage is kept
  as a placeholder; the misleading `__init__.py` comment advertising it has been
  removed).
- `tests/` imports via a `sys.path` hack rather than the installed package.
- `ultrasonic.estimate_bandpass` / `estimate_bandpass_centroid` have placeholder
  (`_summary_`) docstrings.
