"""Tests for the coordinate helpers in utils/coordinates.py.

Checks the analytic anchors and invariants of the coordinate conversions
and spherical grids:

  (1) sph2cart maps the canonical directions (poles, x/y axes) to the
      expected Cartesian unit vectors, and cart2sph inverts sph2cart for
      arbitrary angles (round trip, physics/ISO convention).
  (2) _calc_sample_size implements N ~ 4*pi / theta^2.
  (3) equispaced_S2_grid (Fibonacci sphere / sunflower mapping) returns
      unit vectors whose measured mean spacing matches the requested
      one, hemisphere halves at equal density, and the optional
      coordinate-axis directions.
  (4) equispaced_S2_grid_offset (offset Fibonacci lattice) returns
      angles inside their domains, hemisphere filtering, and the
      degrees flag.

Plus the documented input-validation behaviour (ValueError on malformed
input).

Run standalone (no pytest dependency):
    python tests/test_coordinates.py
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

from pyrockwave.utils.coordinates import (
    sph2cart,
    cart2sph,
    equispaced_S2_grid,
    equispaced_S2_grid_offset,
    _calc_sample_size,
)


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_sph2cart_canonical_directions():
    # north pole, south pole, +x, +y (ISO physics convention)
    x, y, z = sph2cart(0.0, 0.0)
    assert np.allclose((x, y, z), (0, 0, 1))
    x, y, z = sph2cart(0.0, np.pi)
    assert np.allclose((x, y, z), (0, 0, -1))
    x, y, z = sph2cart(0.0, np.pi / 2)
    assert np.allclose((x, y, z), (1, 0, 0))
    x, y, z = sph2cart(np.pi / 2, np.pi / 2)
    assert np.allclose((x, y, z), (0, 1, 0))


def test_sph2cart_cart2sph_roundtrip():
    rng = np.random.default_rng(0)
    azimuth = rng.uniform(0, 2 * np.pi, 500)
    polar = rng.uniform(0.01, np.pi - 0.01, 500)  # avoid poles (azimuth undefined)
    r_in = rng.uniform(0.5, 3.0, 500)

    x, y, z = sph2cart(azimuth, polar, r_in)
    r, theta, phi = cart2sph(x, y, z)

    assert np.allclose(r, r_in)
    assert np.allclose(theta, polar)
    assert np.allclose(phi, azimuth)


def test_cart2sph_angle_ranges():
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(300, 3))
    r, theta, phi = cart2sph(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert np.all(r > 0)
    assert np.all((theta >= 0) & (theta <= np.pi))
    assert np.all((phi >= 0) & (phi < 2 * np.pi))


def test_calc_sample_size():
    assert _calc_sample_size(1.0) == 41_253
    # halving the spacing quadruples the sample size (approximately)
    assert abs(_calc_sample_size(0.5) / _calc_sample_size(1.0) - 4) < 0.01
    assert expect_value_error(_calc_sample_size, 0.0)
    assert expect_value_error(_calc_sample_size, -1.0)


def test_equispaced_S2_grid_unit_sphere_and_size():
    spacing = 5.0
    points = equispaced_S2_grid(ang_spacing_deg=spacing)
    assert points.shape == (_calc_sample_size(spacing), 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    # default spacing is 1 degree
    assert equispaced_S2_grid().shape == (41_253, 3)


def test_equispaced_S2_grid_measured_spacing():
    # the measured mean nearest-neighbour angular distance should be
    # close to the requested spacing
    for spacing in (2.0, 5.0):
        points = equispaced_S2_grid(ang_spacing_deg=spacing)
        chord, _ = cKDTree(points).query(points, k=2)
        angular_deg = np.rad2deg(2 * np.arcsin(chord[:, 1] / 2))
        assert abs(angular_deg.mean() - spacing) / spacing < 0.1


def test_equispaced_S2_grid_hemispheres():
    spacing = 3.0
    n_full = _calc_sample_size(spacing)
    upper = equispaced_S2_grid(ang_spacing_deg=spacing, hemisphere="upper")
    lower = equispaced_S2_grid(ang_spacing_deg=spacing, hemisphere="lower")
    # same density as the full sphere: about half the points per side
    assert abs(len(upper) - n_full / 2) <= 1
    assert abs(len(lower) - n_full / 2) <= 1
    assert np.all(upper[:, 2] >= 0)
    assert np.all(lower[:, 2] <= 0)


def test_equispaced_S2_grid_include_axes():
    spacing = 10.0
    n_full = _calc_sample_size(spacing)
    axes = np.array(
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=float,
    )

    points = equispaced_S2_grid(ang_spacing_deg=spacing, include_axes=True)
    assert points.shape == (n_full + 6, 3)
    assert np.allclose(points[:6], axes)

    upper = equispaced_S2_grid(
        ang_spacing_deg=spacing, hemisphere="upper", include_axes=True
    )
    assert np.allclose(upper[:5], axes[axes[:, 2] >= 0])
    assert np.all(upper[:, 2] >= 0)
    assert not np.any(np.all(np.isclose(upper, [0, 0, -1]), axis=1))

    lower = equispaced_S2_grid(
        ang_spacing_deg=spacing, hemisphere="lower", include_axes=True
    )
    assert np.all(lower[:, 2] <= 0)
    assert not np.any(np.all(np.isclose(lower, [0, 0, 1]), axis=1))


def test_equispaced_S2_grid_rejects_bad_inputs():
    assert expect_value_error(equispaced_S2_grid, ang_spacing_deg=0.0)
    assert expect_value_error(equispaced_S2_grid, ang_spacing_deg=-1.0)
    assert expect_value_error(equispaced_S2_grid, ang_spacing_deg="1")
    assert expect_value_error(equispaced_S2_grid, hemisphere="north")


def test_equispaced_S2_grid_offset_full_sphere():
    spacing = 2.0
    azimuth, polar = equispaced_S2_grid_offset(ang_spacing_deg=spacing)
    assert azimuth.shape == polar.shape == (_calc_sample_size(spacing),)
    assert np.all((azimuth >= 0) & (azimuth < 2 * np.pi))
    assert np.all((polar >= 0) & (polar <= np.pi))
    # sorted by polar angle, with one point at each pole
    assert np.all(np.diff(polar) >= 0)
    assert np.isclose(polar[0], 0) and np.isclose(polar[-1], np.pi)


def test_equispaced_S2_grid_offset_hemispheres_and_degrees():
    azimuth, polar = equispaced_S2_grid_offset(
        ang_spacing_deg=3.0, hemisphere="upper"
    )
    assert np.all(polar <= np.pi / 2 + 1e-12)
    azimuth, polar = equispaced_S2_grid_offset(
        ang_spacing_deg=3.0, hemisphere="lower"
    )
    assert np.all(polar >= np.pi / 2 - 1e-12)
    azimuth, polar = equispaced_S2_grid_offset(ang_spacing_deg=3.0, degrees=True)
    assert np.all(polar <= 180.0) and np.all(azimuth < 360.0)


def test_equispaced_S2_grid_offset_rejects_bad_inputs():
    assert expect_value_error(equispaced_S2_grid_offset, ang_spacing_deg=0.0)
    assert expect_value_error(equispaced_S2_grid_offset, ang_spacing_deg="1")
    assert expect_value_error(equispaced_S2_grid_offset, ang_spacing_deg=150.0)
    assert expect_value_error(equispaced_S2_grid_offset, hemisphere="north")


def main():
    checks = [
        test_sph2cart_canonical_directions,
        test_sph2cart_cart2sph_roundtrip,
        test_cart2sph_angle_ranges,
        test_calc_sample_size,
        test_equispaced_S2_grid_unit_sphere_and_size,
        test_equispaced_S2_grid_measured_spacing,
        test_equispaced_S2_grid_hemispheres,
        test_equispaced_S2_grid_include_axes,
        test_equispaced_S2_grid_rejects_bad_inputs,
        test_equispaced_S2_grid_offset_full_sphere,
        test_equispaced_S2_grid_offset_hemispheres_and_degrees,
        test_equispaced_S2_grid_offset_rejects_bad_inputs,
    ]
    ok = True
    for check in checks:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except AssertionError:
            ok = False
            print(f"[FAIL] {check.__name__}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
