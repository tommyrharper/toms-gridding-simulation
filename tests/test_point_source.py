import numpy as np
import pytest

from gridding_sim.simulate import ARCSEC, field_halfwidth_arcsec, make_point_sources, point_source_vis


def test_point_source_vis_no_sources_is_zero():
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.5, -0.5])
    V = point_source_vis(u, v, [])
    assert np.allclose(V, 0.0)


def test_point_source_vis_centre_source_is_constant_flux():
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.5, -0.5])
    V = point_source_vis(u, v, [(0.0, 0.0, 3.0)])
    assert np.allclose(V, 3.0)


def test_point_source_vis_matches_analytic_fringe():
    l, m, flux = 2.0 * ARCSEC, -1.0 * ARCSEC, 2.0
    u, v = np.array([10.0]), np.array([5.0])
    V = point_source_vis(u, v, [(l, m, flux)])
    expected = flux * np.exp(-2j * np.pi * (u * l + v * m))
    assert np.allclose(V, expected)


def test_point_source_vis_is_linear_superposition():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([-1.0, 0.5, 2.0])
    sources = [(1e-6, 2e-6, 1.5), (-3e-6, 4e-6, 0.7)]
    combined = point_source_vis(u, v, sources)
    separate = sum(point_source_vis(u, v, [s]) for s in sources)
    assert np.allclose(combined, separate)


def test_field_halfwidth_arcsec_formula():
    assert field_halfwidth_arcsec(256, 0.1) == pytest.approx(0.8 * 128 * 0.1)
    assert field_halfwidth_arcsec(100, 1.0, margin=1.0) == pytest.approx(50.0)


def test_make_point_sources_single_mode():
    sources = make_point_sources("single", npix=256, cell_arcsec=0.1, flux=5.0)
    assert sources == [(0.0, 0.0, 5.0)]


def test_make_point_sources_manual_mode_converts_arcsec_to_radians():
    sources = make_point_sources(
        "manual", npix=256, cell_arcsec=0.1, manual=[(1.0, 2.0, 3.0)]
    )
    assert len(sources) == 1
    dl, dm, flux = sources[0]
    assert dl == pytest.approx(1.0 * ARCSEC)
    assert dm == pytest.approx(2.0 * ARCSEC)
    assert flux == 3.0


def test_make_point_sources_manual_mode_defaults_to_empty_list():
    assert make_point_sources("manual", npix=256, cell_arcsec=0.1) == []


def test_make_point_sources_random_mode_respects_count_bounds_and_flux_range():
    npix, cell_arcsec, margin = 100, 1.0, 0.8
    rng = np.random.default_rng(0)
    sources = make_point_sources(
        "random",
        npix,
        cell_arcsec,
        n=25,
        rng=rng,
        flux_range=(0.5, 5.0),
        margin=margin,
    )
    assert len(sources) == 25
    hw = margin * (npix // 2) * cell_arcsec * ARCSEC
    for l, m, flux in sources:
        assert -hw <= l <= hw
        assert -hw <= m <= hw
        assert 0.5 <= flux <= 5.0


def test_make_point_sources_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_point_sources("bogus", npix=100, cell_arcsec=1.0)
