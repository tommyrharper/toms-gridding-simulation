from arrays import CONFIG_DIR, list_arrays


def test_returns_sorted_list_of_config_stems():
    expected = sorted(p.name[:-4] for p in CONFIG_DIR.glob("*.cfg"))
    assert list_arrays() == expected


def test_no_cfg_extension_in_names():
    assert all(not name.endswith(".cfg") for name in list_arrays())


def test_nonempty():
    assert len(list_arrays()) > 0
