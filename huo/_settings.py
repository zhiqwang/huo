"""Framework-independent optimizer configuration checks."""

import math
from numbers import Integral, Real


def validate_optimization(steps, lr, nonnegative):
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not isinstance(nonnegative, bool):
        raise ValueError("nonnegative must be a bool")
