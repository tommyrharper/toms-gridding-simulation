import numpy as np
import pytest

from main import require_visibilities


def test_require_visibilities_passes_when_source_is_up():
    require_visibilities(np.array([1.0, 2.0]), {"max_elev_deg": 45.0}, "vla.a", 34.0)


def test_require_visibilities_raises_when_source_never_rises():
    info = {"max_elev_deg": -12.3}
    with pytest.raises(ValueError, match="vla.a"):
        require_visibilities(np.array([]), info, "vla.a", -80.0)
