import numpy as np
import pytest

from simulate import ARCSEC, dirty_beam, dirty_image, w_term_error


def test_zero_baseline_gives_uniform_image_of_real_part():
    u = np.array([0.0])
    v = np.array([0.0])
    V = np.array([2.0 + 1.0j])
    img = dirty_image(u, v, V, npix=5, cell_arcsec=1.0)
    assert img.shape == (5, 5)
    assert np.allclose(img, 2.0)


def test_zero_baseline_applies_natural_weighting():
    u = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    V = np.array([1.0 + 0j, 3.0 + 0j])
    weights = np.array([1.0, 3.0])
    img = dirty_image(u, v, V, npix=3, cell_arcsec=1.0, weights=weights)
    assert np.allclose(img, 2.5)  # (1*1 + 3*3) / (1 + 3)


def test_duplicating_a_baseline_leaves_natural_weighted_image_unchanged():
    u1, v1, V1 = np.array([12.0]), np.array([-4.0]), np.array([1.5 - 0.5j])
    img1 = dirty_image(u1, v1, V1, npix=9, cell_arcsec=0.3)

    u2 = np.array([12.0, 12.0])
    v2 = np.array([-4.0, -4.0])
    V2 = np.array([1.5 - 0.5j, 1.5 - 0.5j])
    img2 = dirty_image(u2, v2, V2, npix=9, cell_arcsec=0.3)

    assert np.allclose(img1, img2)


def test_dirty_beam_is_dirty_image_of_unit_point_source():
    u = np.array([10.0, -5.0, 3.0])
    v = np.array([3.0, 7.0, -2.0])
    ones = np.ones(u.shape, dtype=np.complex128)
    expected = dirty_image(u, v, ones, npix=8, cell_arcsec=0.5)
    beam = dirty_beam(u, v, npix=8, cell_arcsec=0.5)
    assert np.allclose(beam, expected)


def test_dirty_beam_peaks_at_phase_centre():
    u = np.array([10.0, -5.0, 3.0, 42.0])
    v = np.array([3.0, 7.0, -2.0, -18.0])
    npix = 7  # odd -> exact zero-coordinate pixel at the centre
    beam = dirty_beam(u, v, npix=npix, cell_arcsec=0.5)
    centre = npix // 2
    assert beam[centre, centre] == pytest.approx(1.0, abs=1e-6)
    assert beam[centre, centre] == pytest.approx(beam.max(), abs=1e-6)


def test_w_term_error_is_zero_for_zero_w():
    assert w_term_error(np.array([0.0]), npix=100, cell_arcsec=1.0) == 0.0


def test_w_term_error_matches_formula():
    w = np.array([1000.0, -500.0])
    npix, cell_arcsec = 100, 1.0
    theta_edge = (npix // 2) * cell_arcsec * ARCSEC
    expected = 2 * np.pi * np.abs(w).max() * (1.0 - np.cos(theta_edge))
    assert w_term_error(w, npix, cell_arcsec) == pytest.approx(expected)


def test_w_term_error_grows_with_field_of_view():
    w = np.array([1000.0])
    small = w_term_error(w, npix=20, cell_arcsec=1.0)
    large = w_term_error(w, npix=2000, cell_arcsec=1.0)
    assert large > small
