import numpy as np
from typing import Iterator

def mlcg(seed: int = 123456789, a: int = 1664525, c: int = 1013904223, m: int = 2**32) -> Iterator[float]:
    """
    Generador congruencial lineal multiplicativo (estilo Numerical Recipes).
    Devuelve valores en [0,1).
    """
    state = seed % m
    while True:
        state = (a * state + c) % m
        yield state / m

def mlcg_sample(n: int, seed: int = 123456789) -> np.ndarray:
    g = mlcg(seed=seed)
    return np.fromiter((next(g) for _ in range(n)), dtype=float, count=n)