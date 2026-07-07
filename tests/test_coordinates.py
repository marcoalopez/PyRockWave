"""Tests for the coordinate helpers in utils/coordinates.py.

Checks the analytic anchors and invariants of the coordinate conversions
and spherical grids:

  (1) sph2cart maps the canonical directions (poles, x/y axes) to the
      expected Cartesian unit vectors, and cart2sph inverts sph2cart for
      arbitrary angles (round trip, physics/ISO convention).
  (2) equispaced_S2_grid returns the requested number of points, angles
      inside their domains, hemisphere filtering, and the degrees flag.
  (3) equispaced_S2_grid_fsa returns points on the unit sphere.

Plus the documented input-validation behaviour (ValueError on malformed
input).

Run standalone (no pytest dependency):
    python tests/test_coordinates.py
"""

import sys

import numpy as np

from pyrockwave.utils.coordinates import (
    sph2cart,
    cart2sph,
    equispaced_S2_grid,
    equispaced_S2_grid_fsa,
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


def test_equispaced_S2_grid_full_sphere():
    n = 1000
    azimuth, polar = equispaced_S2_grid(num_points=n)
    assert azimuth.shape == polar.shape == (n,)
    assert np.all((azimuth >= 0) & (azimuth < 2 * np.pi))
    assert np.all((polar >= 0) & (polar <= np.pi))
    # sorted by polar angle, with one point at each pole
    assert np.all(np.diff(polar) >= 0)
    assert np.isclose(polar[0], 0) and np.isclose(polar[-1], np.pi)


def test_equispaced_S2_grid_hemispheres_and_degrees():
    azimuth, polar = equispaced_S2_grid(num_points=500, hemisphere="upper")
    assert np.all(polar <= np.pi / 2 + 1e-12)
    azimuth, polar = equispaced_S2_grid(num_points=500, hemisphere="lower")
    assert np.all(polar >= np.pi / 2 - 1e-12)
    azimuth, polar = equispaced_S2_grid(num_points=500, degrees=True)
    assert np.all(polar <= 180.0) and np.all(azimuth < 360.0)


def test_equispaced_S2_grid_rejects_bad_inputs():
    assert expect_value_error(equispaced_S2_grid, num_points=0)
    assert expect_value_error(equispaced_S2_grid, num_points=10.5)
    assert expect_value_error(equispaced_S2_grid, hemisphere="north")


def test_equispaced_S2_grid_fsa_unit_sphere():
    n = 250
    points = equispaced_S2_grid_fsa(n)
    assert points.shape == (n, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    assert expect_value_error(equispaced_S2_grid_fsa, 0)


def main():
    checks = [
        test_sph2cart_canonical_directions,
        test_sph2cart_cart2sph_roundtrip,
        test_cart2sph_angle_ranges,
        test_equispaced_S2_grid_full_sphere,
        test_equispaced_S2_grid_hemispheres_and_degrees,
        test_equispaced_S2_grid_rejects_bad_inputs,
        test_equispaced_S2_grid_fsa_unit_sphere,
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
