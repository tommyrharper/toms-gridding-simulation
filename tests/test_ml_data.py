import numpy as np
import torch

from ml.data import N_FEATURES, make_example, pad_visibilities
from gridding_sim.simulate import dirty_image, point_source_vis


def test_pad_visibilities_shape_and_dtype():
    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0])
    V = np.array([1 + 2j, 3 - 1j])
    x = pad_visibilities(u, v, V, max_vis=5)
    assert x.shape == (5, N_FEATURES)
    assert x.dtype == torch.float32


def test_pad_visibilities_masks_real_and_padded_rows():
    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0])
    V = np.array([1 + 2j, 3 - 1j])
    x = pad_visibilities(u, v, V, max_vis=4)

    assert torch.allclose(x[0], torch.tensor([1.0, 3.0, 1.0, 2.0, 1.0]))
    assert torch.allclose(x[1], torch.tensor([2.0, 4.0, 3.0, -1.0, 1.0]))
    assert torch.all(x[2:] == 0.0)


def test_pad_visibilities_truncates_deterministically_when_oversized():
    u = np.arange(5.0)
    v = np.arange(5.0)
    V = np.arange(5.0) + 0j
    x = pad_visibilities(u, v, V, max_vis=3)
    assert x.shape == (3, N_FEATURES)
    assert torch.all(x[:, 4] == 1.0)
    assert torch.allclose(x[:, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_pad_visibilities_empty_input_is_all_zero_no_nan_inf():
    u = np.array([])
    v = np.array([])
    V = np.array([], dtype=np.complex128)
    x = pad_visibilities(u, v, V, max_vis=6)
    assert x.shape == (6, N_FEATURES)
    assert torch.all(x == 0.0)
    assert torch.all(torch.isfinite(x))


def test_pad_visibilities_is_deterministic():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    V = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    x1 = pad_visibilities(u, v, V, max_vis=8)
    x2 = pad_visibilities(u, v, V, max_vis=8)
    assert torch.equal(x1, x2)


def test_make_example_target_matches_dirty_image_ground_truth():
    rng = np.random.default_rng(0)
    u = rng.uniform(-50, 50, 20)
    v = rng.uniform(-50, 50, 20)
    sources = [(0.0, 0.0, 1.0)]
    npix, cell = 8, 1.0

    x, y = make_example(u, v, sources, max_vis=32, npix=npix, cell_arcsec=cell)

    V = point_source_vis(u, v, sources)
    expected = dirty_image(u, v, V, npix, cell).astype(np.float32).ravel()

    assert x.shape == (32, N_FEATURES)
    assert y.shape == (npix * npix,)
    assert y.dtype == torch.float32
    assert np.allclose(y.numpy(), expected)


def test_make_example_target_matches_truncated_input_when_oversized():
    rng = np.random.default_rng(1)
    u = rng.uniform(-50, 50, 20)
    v = rng.uniform(-50, 50, 20)
    sources = [(0.0, 0.0, 1.0)]
    npix, cell = 8, 1.0
    max_vis = 10

    x, y = make_example(u, v, sources, max_vis=max_vis, npix=npix, cell_arcsec=cell)

    V = point_source_vis(u[:max_vis], v[:max_vis], sources)
    expected = dirty_image(u[:max_vis], v[:max_vis], V, npix, cell).astype(np.float32).ravel()

    assert x.shape == (max_vis, N_FEATURES)
    assert np.allclose(y.numpy(), expected)
