import math

import pytest

from huo import RadonFanbeam


@pytest.fixture
def radon():
    return RadonFanbeam(
        resolution=16,
        angles=[index * math.pi / 6 for index in range(12)],
        source_distance=109,
        det_distance=73 / 3,
        det_count=20,
        det_spacing=1,
    )
