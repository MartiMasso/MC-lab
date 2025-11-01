import numpy as np

def sum12_gaussian(n: int, rng=np.random) -> np.ndarray:
    """Aproximación N(0,1) sumando 12 uniformes y restando 6."""
    return rng.random((n, 12)).sum(axis=1) - 6.0

def box_muller(n: int, rng=np.random) -> np.ndarray:
    """Devuelve n muestras ~ N(0,1) con Box-Muller (pares)."""
    m = (n + 1) // 2  # número de pares
    u1 = rng.random(m)
    u2 = rng.random(m)
    r = np.sqrt(-2.0 * np.log(u1 + 1e-15))
    theta = 2.0 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    z = np.concatenate([z1, z2])[:n]
    return z