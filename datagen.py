import numpy as np


def generate_data(
    a: float,
    b: float,
    c: float,
    N: int,
    sigma: float,
    low: float = 0,
    high: float = 2,
):
    z = np.random.uniform(low, high, N)
    eps = np.random.normal(0, sigma, N)
    return z, z**2 * a + z * b + c + eps
