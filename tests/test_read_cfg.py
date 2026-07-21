import numpy as np

from arrays import CONFIG_DIR, _read_cfg


def write_cfg(tmp_path, text):
    path = tmp_path / "test.cfg"
    path.write_text(text)
    return path


def test_parses_positions_and_headers(tmp_path):
    path = write_cfg(
        tmp_path,
        "# observatory=VLA\n"
        "# coordsys=XYZ\n"
        "-1601614.061201\t-5042001.676547\t3554652.455603\t25.  W08\n"
        "-1602592.823528\t-5042055.013423\t3554140.652770\t25.  W16\n",
    )
    xyz, coordsys, obs = _read_cfg(path)
    assert coordsys == "XYZ"
    assert obs == "VLA"
    assert np.allclose(
        xyz,
        [
            [-1601614.061201, -5042001.676547, 3554652.455603],
            [-1602592.823528, -5042055.013423, 3554140.652770],
        ],
    )


def test_parses_loc_coordsys_with_trailing_text(tmp_path):
    path = write_cfg(
        tmp_path,
        "# observatory=ACA\n"
        "# coordsys=LOC (local tangent plane)  \n"
        "-8.494703874\t155.5405059\t-2.091543332\t7.   J501\n",
    )
    xyz, coordsys, obs = _read_cfg(path)
    assert coordsys == "LOC"
    assert obs == "ACA"
    assert xyz.shape == (1, 3)


def test_defaults_when_no_header_comments(tmp_path):
    path = write_cfg(tmp_path, "1.0 2.0 3.0\n4.0 5.0 6.0\n")
    xyz, coordsys, obs = _read_cfg(path)
    assert coordsys == "XYZ"
    assert obs == ""
    assert xyz.shape == (2, 3)


def test_ignores_blank_lines_and_unrelated_comments(tmp_path):
    path = write_cfg(
        tmp_path,
        "# x y z diam pad#\n"
        "\n"
        "1.0 2.0 3.0 25. W08\n"
        "\n"
        "4.0 5.0 6.0 25. W16\n",
    )
    xyz, coordsys, obs = _read_cfg(path)
    assert xyz.shape == (2, 3)
    assert coordsys == "XYZ"
    assert obs == ""


def test_no_data_rows_returns_empty_array(tmp_path):
    path = write_cfg(tmp_path, "# observatory=VLA\n# coordsys=XYZ\n")
    xyz, coordsys, obs = _read_cfg(path)
    assert xyz.shape == (0,)
    assert coordsys == "XYZ"
    assert obs == "VLA"


def test_real_vla_config_file():
    xyz, coordsys, obs = _read_cfg(CONFIG_DIR / "vla.a.cfg")
    assert coordsys == "XYZ"
    assert obs == "VLA"
    assert xyz.ndim == 2
    assert xyz.shape[1] == 3
    assert xyz.shape[0] > 0


def test_real_aca_loc_config_file():
    xyz, coordsys, obs = _read_cfg(CONFIG_DIR / "aca.all.cfg")
    assert coordsys == "LOC"
    assert obs == "ACA"
    assert xyz.shape[1] == 3
