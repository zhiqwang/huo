import math

import pytest

from huo import FanbeamGeometry


@pytest.fixture
def projector():
    pytest.importorskip("torch")
    from huo.torch import FanbeamProjector

    geometry = FanbeamGeometry(
        resolution=16,
        source_distance=109,
        det_distance=73 / 3,
        det_count=20,
        det_spacing=1,
    )
    return FanbeamProjector(geometry, [index * math.pi / 6 for index in range(12)])
