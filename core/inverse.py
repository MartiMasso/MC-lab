import numpy as np

def sample_exponential(tau: float, n: int, rng=np.random) -> np.ndarray:
    """
    x = -tau * ln(1-u), u~U(0,1)
    """
    u = rng.random(n)
    return -tau * np.log(1.0 - u + 1e-15)