import numpy as np
import pytest

from arrays import (
    CONFIG_DIR,
    _LOC_SITE,
    _enu_to_local_equatorial,
    antennas_local_equatorial,
)


def test_enu_to_local_equatorial_at_equator():
    enu = np.array([[1.0, 2.0, 3.0]])
    out = _enu_to_local_equatorial(enu, lat=0.0)
    assert np.allclose(out, [[3.0, 1.0, 2.0]])


def test_enu_to_local_equatorial_at_pole():
    enu = np.array([[1.0, 2.0, 3.0]])
    out = _enu_to_local_equatorial(enu, lat=np.pi / 2)
    assert np.allclose(out, [[-2.0, 1.0, 3.0]])


def test_enu_to_local_equatorial_preserves_norm():
    rng = np.random.default_rng(0)
    enu = rng.normal(size=(10, 3))
    for lat_deg in (-60, -30, 0, 15, 45, 89):
        out = _enu_to_local_equatorial(enu, lat=np.deg2rad(lat_deg))
        assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(enu, axis=1))


def test_antennas_local_equatorial_xyz_config():
    le, lat, lon = antennas_local_equatorial("vla.a")
    assert le.shape[1] == 3
    # VLA site: ~34.08 N, ~107.62 W
    assert np.rad2deg(lat) == pytest.approx(34.08, abs=0.5)
    assert lon == pytest.approx(-107.62, abs=0.5)


def test_antennas_local_equatorial_loc_config_uses_known_site():
    le, lat, lon = antennas_local_equatorial("aca.all")
    lat_deg, lon_deg = _LOC_SITE["ACA"]
    assert np.rad2deg(lat) == pytest.approx(lat_deg)
    assert lon == pytest.approx(lon_deg)
    assert le.shape[1] == 3


def test_antennas_local_equatorial_unknown_loc_site_raises(tmp_path):
    cfg = tmp_path / "unknownsite.cfg"
    cfg.write_text(
        "# observatory=NOTAKNOWNSITE\n"
        "# coordsys=LOC\n"
        "1.0 2.0 3.0\n"
    )
    with pytest.raises(NotImplementedError):
        antennas_local_equatorial(str(cfg))


def test_antennas_local_equatorial_unsupported_coordsys_raises(tmp_path):
    cfg = tmp_path / "weirdsys.cfg"
    cfg.write_text(
        "# observatory=VLA\n"
        "# coordsys=WEIRD\n"
        "1.0 2.0 3.0\n"
    )
    with pytest.raises(NotImplementedError):
        antennas_local_equatorial(str(cfg))
