import sys

sys.path.append(".")

import pytest
import numpy as np
from utils.resource_timeline import ResourceTimeline


def test_resource_timeline():
    rt = ResourceTimeline(4, True)
    a0 = rt.availability(2)
    assert a0 == (0, None, None)
    rt.consume(1, 2, 0, 3)
    a1 = rt.availability(2)
    assert a1 == (0, None, None)
    rt.consume(2, 2, 0, 4)
    a2 = rt.availability(2)
    assert a2 == (3, 1, False)
    rt.consume(3, 2, 4, 10)
    rt.consume(4, 2, 4, 6)
    a3 = rt.availability(3)
    assert a3 == (10, 3, False)
    rt.consume(5, 3, 10, 11)
    a4 = rt.availability(3)
    assert a4 == (11, 5, False)
    rt.consume(6, 3, 11, 15)
    a5 = rt.availability(2)
    assert a5 == (15, 6, False)
