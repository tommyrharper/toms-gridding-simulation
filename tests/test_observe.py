import numpy as np
import astropy.units as u
import pytest
from astropy.coordinates import Angle
from astropy.time import Time

from arrays import antennas_local_equatorial
from observe import _transit_centered_start, observe


def test_transit_centered_start_puts_ha_at_zero_at_midpoint():
    lon_deg = -107.618
    ra_deg = 45.0
    duration_h = 2.0
    start = _transit_centered_start(ra_deg, duration_h, lon_deg)
    midpoint = start + (duration_h / 2) * u.hour
    lst = midpoint.sidereal_time("apparent", longitude=lon_deg * u.deg)
    ha = Angle(lst - ra_deg * u.deg).wrap_at(12 * u.hourangle)
    assert ha.hourangle == pytest.approx(0.0, abs=1e-6)


def test_transit_centered_start_returns_time():
    start = _transit_centered_start(0.0, 1.0, 0.0)
    assert isinstance(start, Time)


def test_observe_near_zenith_source_reaches_near_90_deg_elevation():
    _, lat, _ = antennas_local_equatorial("vla.a")
    dec_deg = np.rad2deg(lat)  # source that transits at VLA's zenith
    u_, v_, w_, info = observe(
        ra_deg=0.0, dec_deg=dec_deg, duration_h=0.05, integration_s=30, array="vla.a"
    )
    assert info["max_elev_deg"] == pytest.approx(90.0, abs=0.01)
    assert u_.size == v_.size == w_.size == info["n_vis"]


def test_observe_baseline_and_vis_counts_match_antenna_count():
    n_antennas = antennas_local_equatorial("vla.a")[0].shape[0]
    expected_baselines = n_antennas * (n_antennas - 1) // 2
    u_, v_, w_, info = observe(
        ra_deg=0.0, dec_deg=34.0, duration_h=0.5, integration_s=60, array="vla.a"
    )
    assert info["n_baseline"] == expected_baselines
    assert info["n_vis"] == info["n_baseline"] * info["n_dump_up"]


def test_observe_source_that_never_rises_returns_empty_arrays():
    u_, v_, w_, info = observe(
        ra_deg=0.0, dec_deg=-89.0, duration_h=2.0, integration_s=60, array="vla.a"
    )
    assert u_.size == 0
    assert v_.size == 0
    assert w_.size == 0
    assert info["n_dump_up"] == 0
    assert info["n_vis"] == 0
    assert info["ha_up_hours"] is None
    assert info["w_range"] is None


def test_observe_dump_count_scales_with_integration_time():
    _, _, _, info_coarse = observe(
        ra_deg=0.0, dec_deg=34.0, duration_h=1.0, integration_s=60, array="vla.a"
    )
    _, _, _, info_fine = observe(
        ra_deg=0.0, dec_deg=34.0, duration_h=1.0, integration_s=30, array="vla.a"
    )
    assert info_fine["n_dump"] == 2 * info_coarse["n_dump"]


def test_observe_respects_explicit_start_time():
    start = "2026-06-01T00:00:00"
    u_, v_, w_, info = observe(
        ra_deg=0.0,
        dec_deg=34.0,
        start=start,
        duration_h=0.5,
        integration_s=60,
        array="vla.a",
    )
    assert info["n_dump"] == 30
