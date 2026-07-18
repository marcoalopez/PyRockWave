"""Tests for the plotting helpers in plotting/plots.py.

Checks the analytic anchors of the visibility culling used for 3D
sphere plots:

  (1) visible_mask keeps points facing the camera and drops points on
      the far hemisphere, for known view directions.
  (2) culled_quiver draws exactly the arrows anchored on the visible
      hemisphere (one 3-segment line per arrow with arrow_length_ratio=0).

Plus the documented input-validation behaviour (ValueError on malformed
input).

Run standalone (no pytest dependency):
    python tests/test_plots.py
"""

import sys

import matplotlib

matplotlib.use("Agg")  # headless backend for testing

import matplotlib.pyplot as plt
import numpy as np

from pyrockwave.plotting.plots import culled_quiver, visible_mask


def expect_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def make_ax(elev, azim):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


def test_visible_mask_known_views():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    # camera on +x: only +x visible (equator/limb points trimmed by margin)
    fig, ax = make_ax(elev=0, azim=0)
    assert np.array_equal(
        visible_mask(points, ax), [True, False, False, False, False, False]
    )
    plt.close(fig)

    # camera on +z (looking down): only +z visible
    fig, ax = make_ax(elev=90, azim=0)
    assert np.array_equal(
        visible_mask(points, ax), [False, False, False, False, True, False]
    )
    plt.close(fig)

    # camera on +y
    fig, ax = make_ax(elev=0, azim=90)
    assert np.array_equal(
        visible_mask(points, ax), [False, False, True, False, False, False]
    )
    plt.close(fig)


def test_visible_mask_hemisphere_split():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(2000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    fig, ax = make_ax(elev=35, azim=-60)
    mask = visible_mask(points, ax, margin=0.0)
    # with zero margin, close to half of a random sample is visible
    assert abs(mask.mean() - 0.5) < 0.05
    # larger margins can only shrink the visible set
    assert np.all(visible_mask(points, ax, margin=0.1) <= mask)
    plt.close(fig)


def test_culled_quiver_draws_visible_arrows_only():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(300, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    vectors = rng.normal(size=(300, 3))

    fig, ax = make_ax(elev=30, azim=45)
    n_visible = int(visible_mask(points, ax).sum())
    artist = culled_quiver(
        ax, points, vectors, pivot="middle", length=0.15, arrow_length_ratio=0
    )
    # with arrow_length_ratio=0 each arrow is drawn as 3 segments
    # (shaft + two zero-length head lines)
    assert len(artist.get_segments()) == 3 * n_visible
    plt.close(fig)


def test_rejects_bad_inputs():
    fig, ax = make_ax(elev=0, azim=0)
    good = np.zeros((5, 3))

    assert expect_value_error(visible_mask, np.zeros((5, 2)), ax)
    assert expect_value_error(visible_mask, [[1, 0, 0]], ax)
    assert expect_value_error(culled_quiver, ax, good, np.zeros((4, 3)))
    assert expect_value_error(culled_quiver, ax, good, None)
    plt.close(fig)


def main():
    checks = [
        test_visible_mask_known_views,
        test_visible_mask_hemisphere_split,
        test_culled_quiver_draws_visible_arrows_only,
        test_rejects_bad_inputs,
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
