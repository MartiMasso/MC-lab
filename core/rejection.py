import numpy as np
from typing import Callable, Tuple

def sample_rejection(
    f: Callable[[np.ndarray], np.ndarray],
    a: float, b: float, c: float,
    n: int, rng=np.random
) -> Tuple[np.ndarray, float]:
    """
    Muestreo por rechazo (Von Neumann) para una f acotada por c en [a,b].
    Devuelve (muestras, tasa_de_aceptación).
    """
    xs, accepted = [], 0
    trials = 0
    while len(xs) < n:
        x = rng.uniform(a, b)
        y = rng.uniform(0.0, c)
        if y < f(np.array([x]))[0]:
            xs.append(x)
            accepted += 1
        trials += 1
    return np.array(xs), accepted / max(trials, 1)