"""TDD tests for ``grid_visibilities``.

Implement the function in ``gridding_sim.gridtools`` until these pass.
A simple box / tent kernel is injected in most tests so you are not
debugging the spheroidal at the same time as the deposition loop.
"""

import numpy as np
import pytest

from gridding_sim.gridtools import grid_visibilities, spheroidal_gridder
from gridding_sim.simulate import ARCSEC


def _box_kernel(nu, W=6):
    """Nearest-cell kernel: 1 inside |nu| < 0.5, else 0."""
    nu = np.asarray(nu, float)
    return (np.abs(nu) < 0.5).astype(float)


def _tent_kernel(nu, W=6):
    """Linear tent: max(0, 1 - |nu|), independent of W (support ~ 2 cells)."""
    nu = np.asarray(nu, float)
    return np.maximum(0.0, 1.0 - np.abs(nu))


def _du(npix, cell):
    return 1.0 / (npix * cell * ARCSEC)


# ---------------------------------------------------------------------------
# Shape / dtype / empty input
# ---------------------------------------------------------------------------
def test_returns_complex_grid_of_shape_npix():
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=8, cell=1.0, kernel=_box_kernel, W=6)
    assert grid.shape == (8, 8)
    assert np.iscomplexobj(grid)


def test_empty_visibilities_give_zero_grid():
    u = np.array([])
    v = np.array([])
    V = np.array([], dtype=np.complex128)
    grid = grid_visibilities(u, v, V, npix=4, cell=1.0, kernel=_box_kernel, W=6)
    assert grid.shape == (4, 4)
    assert np.allclose(grid, 0.0)


# ---------------------------------------------------------------------------
# Stepping stone: centred index before du / kernel wiring
# ---------------------------------------------------------------------------
def test_single_visibility_at_origin_hardcoded_centre_deposit():
    """u=v=0 on an odd grid: centre pixel gets V (hardcode-friendly first step)."""
    npix = 9
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([2.0 + 1.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=1.0, kernel=_box_kernel, W=6)
    c = npix // 2
    assert grid[c, c] == pytest.approx(2.0 + 1.0j)


# ---------------------------------------------------------------------------
# Centred indexing: index i <-> (i - npix//2) * du
# ---------------------------------------------------------------------------
def test_visibility_at_origin_with_box_kernel_lands_only_on_centre_pixel():
    npix, cell = 9, 1.0  # odd → exact centre index
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([3.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=_box_kernel, W=6)
    c = npix // 2
    assert grid[c, c] == pytest.approx(3.0 + 0.0j)
    grid[c, c] = 0
    assert np.allclose(grid, 0.0)


def test_shift_by_one_uv_cell_moves_deposit_by_one_pixel():
    npix, cell = 9, 0.5
    du = _du(npix, cell)
    c = npix // 2

    # At u = du, v = 0 → centre of pixel (c+1, c) with a box kernel
    u = np.array([du])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=_box_kernel, W=6)

    assert grid[c + 1, c] == pytest.approx(1.0 + 0.0j)
    grid[c + 1, c] = 0
    assert np.allclose(grid, 0.0)


# ---------------------------------------------------------------------------
# Separable kernel product  kernel(nu) * kernel(nv)
# ---------------------------------------------------------------------------
def test_tent_kernel_separable_product_at_half_cell_offsets():
    """Visibility at exact centre; tent spreads to neighbours as k(di)*k(dj)."""
    npix, cell = 7, 1.0
    c = npix // 2
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=_tent_kernel, W=6)

    # Relative to centre: offsets di, dj in {-1,0,1} get (1-|di|)*(1-|dj|)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            expected = (1.0 - abs(di)) * (1.0 - abs(dj))
            assert grid[c + di, c + dj] == pytest.approx(expected)

    # Outside the tent support → ~0
    assert abs(grid[c + 2, c]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Linearity (no normalisation inside grid_visibilities)
# ---------------------------------------------------------------------------
def test_scales_linearly_with_visibility_amplitude():
    npix, cell = 8, 1.0
    u = np.array([0.0])
    v = np.array([0.0])
    g1 = grid_visibilities(u, v, np.array([1.0 + 0.0j]), npix, cell, _box_kernel)
    g2 = grid_visibilities(u, v, np.array([2.5 - 1.0j]), npix, cell, _box_kernel)
    assert np.allclose(g2, (2.5 - 1.0j) * g1)


def test_is_additive_over_visibilities():
    npix, cell = 8, 1.0
    du = _du(npix, cell)
    u = np.array([0.0, du])
    v = np.array([0.0, 0.0])
    V = np.array([1.0 + 0.0j, 2.0 + 0.0j])

    together = grid_visibilities(u, v, V, npix, cell, _box_kernel)
    part0 = grid_visibilities(u[:1], v[:1], V[:1], npix, cell, _box_kernel)
    part1 = grid_visibilities(u[1:], v[1:], V[1:], npix, cell, _box_kernel)
    assert np.allclose(together, part0 + part1)


# ---------------------------------------------------------------------------
# Bounds: no wrap — deposits that fall outside the array are dropped
# ---------------------------------------------------------------------------
def test_visibility_far_outside_grid_does_not_wrap():
    npix, cell = 8, 1.0
    du = _du(npix, cell)
    # Many cells beyond the edge
    u = np.array([(npix + 50) * du])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=_box_kernel, W=6)
    assert np.allclose(grid, 0.0)


# ---------------------------------------------------------------------------
# Real spheroidal kernel smoke test (support / centre)
# ---------------------------------------------------------------------------
def test_spheroidal_at_origin_peaks_at_centre_and_respects_support():
    npix, cell, W = 16, 0.2, 6
    c = npix // 2
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=spheroidal_gridder, W=W)

    k0 = float(spheroidal_gridder(0.0, W=W))
    assert grid[c, c] == pytest.approx(k0 * k0)

    # Outside half-support W/2 = 3 cells from the continuous position, kernel is 0
    assert abs(grid[c + 4, c]) == pytest.approx(0.0, abs=1e-12)
    assert abs(grid[c, c + 4]) == pytest.approx(0.0, abs=1e-12)
    # Inside support should be nonzero
    assert abs(grid[c + 1, c]) > 0.0


# ---------------------------------------------------------------------------
# Independent brute-force reference: for every (i, j) in the grid — not just
# a bounding box around each visibility — accumulate V[k] * kernel(nu) *
# kernel(nv) directly from the documented contract. This exercises the
# floor/ceil bounding-box optimisation in grid_visibilities against a
# structurally different implementation of the same contract.
# ---------------------------------------------------------------------------
def _reference_grid(u, v, V, npix, cell, kernel, W=6):
    du = _du(npix, cell)
    c = npix // 2
    G = np.zeros((npix, npix), dtype=np.complex128)
    for k in range(len(V)):
        for i in range(npix):
            nu = u[k] / du - (i - c)
            wu = kernel(nu, W=W)
            for j in range(npix):
                nv = v[k] / du - (j - c)
                wv = kernel(nv, W=W)
                G[i, j] += V[k] * wu * wv
    return G


@pytest.mark.parametrize("npix", [7, 8, 9, 12, 13])
@pytest.mark.parametrize("cell", [0.05, 0.3, 1.0, 2.5])
@pytest.mark.parametrize(
    "kernel_name,kernel", [("box", _box_kernel), ("tent", _tent_kernel), ("spheroidal", spheroidal_gridder)]
)
def test_matches_brute_force_reference_for_random_visibilities(npix, cell, kernel_name, kernel):
    """Randomised visibilities (including ones straddling/outside the grid)
    must match a from-scratch reference across grid sizes, cell sizes,
    kernel supports, and parities."""
    rng = np.random.default_rng(hash((npix, cell, kernel_name)) & 0xFFFFFFFF)
    du = _du(npix, cell)
    for W in (2, 4, 6, 7):
        n_vis = rng.integers(1, 8)
        u = rng.uniform(-npix, npix, n_vis) * du
        v = rng.uniform(-npix, npix, n_vis) * du
        V = rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
        got = grid_visibilities(u, v, V, npix, cell, kernel, W=W)
        ref = _reference_grid(u, v, V, npix, cell, kernel, W=W)
        assert np.allclose(got, ref, atol=1e-8)


@pytest.mark.parametrize("npix", [8, 9])
def test_visibility_exactly_at_edge_index_matches_reference(npix):
    """Footprint straddling the array boundary — the in-bounds half of the
    kernel support must still deposit correctly (no off-by-one at i=0/npix-1)."""
    du = _du(npix, 1.0)
    c = npix // 2
    for edge_i in (0, npix - 1):
        u = np.array([(edge_i - c) * du])
        v = np.array([0.0])
        V = np.array([1.0 + 0.0j])
        got = grid_visibilities(u, v, V, npix, 1.0, _box_kernel, W=6)
        ref = _reference_grid(u, v, V, npix, 1.0, _box_kernel, W=6)
        assert np.allclose(got, ref)
        assert got[edge_i, c] == pytest.approx(1.0 + 0.0j)


def test_visibility_far_negative_outside_grid_does_not_wrap():
    """Mirror of the existing far-outside test, on the negative side."""
    npix, cell = 8, 1.0
    du = _du(npix, cell)
    u = np.array([-(npix + 50) * du])
    v = np.array([0.0])
    V = np.array([1.0 + 0.0j])
    grid = grid_visibilities(u, v, V, npix=npix, cell=cell, kernel=_box_kernel, W=6)
    assert np.allclose(grid, 0.0)


def test_overlapping_visibilities_accumulate_on_shared_pixels():
    """Two visibilities whose kernel footprints both cover the same pixel
    must sum there, not overwrite one another."""
    npix, cell = 9, 1.0
    c = npix // 2
    u = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])  # identical position -> full overlap
    V = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    grid = grid_visibilities(u, v, V, npix, cell, _box_kernel, W=6)
    assert grid[c, c] == pytest.approx(4.0 + 1.0j)


# ---------------------------------------------------------------------------
# Input validation contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(u=np.array([[0.0]]), v=np.array([0.0]), V=np.array([1.0 + 0j])), "one-dimensional"),
        (dict(u=np.array([0.0, 1.0]), v=np.array([0.0]), V=np.array([1.0 + 0j])), "same length"),
        (dict(npix=0), "npix must be positive"),
        (dict(npix=-3), "npix must be positive"),
        (dict(cell=0.0), "cell must be positive"),
        (dict(cell=-1.0), "cell must be positive"),
        (dict(W=0), "W must be positive"),
        (dict(W=-2), "W must be positive"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs, match):
    defaults = dict(
        u=np.array([0.0]), v=np.array([0.0]), V=np.array([1.0 + 0j]), npix=8, cell=1.0, W=6
    )
    args = {**defaults, **kwargs}
    with pytest.raises(ValueError, match=match):
        grid_visibilities(args["u"], args["v"], args["V"], args["npix"], args["cell"], _box_kernel, W=args["W"])


# ---------------------------------------------------------------------------
# Robustness: a kernel returning a non-0-d array (e.g. shape (1,)) used to
# crash `G[i, j] += ...` with "only 0-dimensional arrays can be converted to
# Python scalars" — grid_visibilities now coerces kernel output to a scalar.
# ---------------------------------------------------------------------------
def test_kernel_returning_one_element_array_does_not_crash():
    def _box_kernel_1d_output(nu, W=6):
        val = 1.0 if abs(float(nu)) < 0.5 else 0.0
        return np.array([val])  # shape (1,), not 0-d

    npix = 8
    c = npix // 2
    grid = grid_visibilities(
        np.array([0.0]), np.array([0.0]), np.array([1.0 + 0.0j]), npix, 1.0, _box_kernel_1d_output, W=6
    )
    assert grid[c, c] == pytest.approx(1.0 + 0.0j)


# ---------------------------------------------------------------------------
# dtype / input-type robustness
# ---------------------------------------------------------------------------
def test_accepts_plain_python_lists():
    grid = grid_visibilities([0.0, 1.0], [0.0, 0.0], [1 + 0j, 2 + 0j], npix=8, cell=1.0, kernel=_box_kernel)
    assert np.iscomplexobj(grid)
    assert grid.sum() == pytest.approx(3 + 0j)


def test_real_valued_visibilities_are_treated_as_zero_imaginary():
    npix = 8
    c = npix // 2
    grid = grid_visibilities(
        np.array([0.0]), np.array([0.0]), np.array([3.0]), npix, 1.0, _box_kernel
    )
    assert grid[c, c] == pytest.approx(3.0 + 0.0j)
