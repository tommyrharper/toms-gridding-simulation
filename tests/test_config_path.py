import pytest

from gridding_sim.arrays import CONFIG_DIR, config_path


def test_resolves_name_without_extension():
    assert config_path("vla.a") == CONFIG_DIR / "vla.a.cfg"


def test_resolves_name_with_extension():
    assert config_path("vla.a.cfg") == CONFIG_DIR / "vla.a.cfg"


def test_resolves_existing_full_path():
    full_path = CONFIG_DIR / "vla.a.cfg"
    assert config_path(str(full_path)) == full_path


def test_raises_for_unknown_config():
    with pytest.raises(FileNotFoundError):
        config_path("not-a-real-array")


def test_raises_for_nonexistent_full_path(tmp_path):
    missing = tmp_path / "missing.cfg"
    with pytest.raises(FileNotFoundError):
        config_path(str(missing))
