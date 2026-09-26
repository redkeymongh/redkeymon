import math
import secrets


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u


def exponentialdist(lam: float) -> float:
    if lam <= 0:
        raise ValueError("lambda must be > 0")
    y = uniform(0.0, 1.0)
    while y == 0.0:
        y = uniform(0.0, 1.0)
    return -math.log(y) / lam
