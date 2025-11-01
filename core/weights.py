import numpy as np
from typing import Tuple

def weighted_histogram(x: np.ndarray, w: np.ndarray, bins: int = 50, range=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, edges = np.histogram(x, bins=bins, range=range, weights=w)
    H2, _     = np.histogram(x, bins=bins, range=range, weights=w**2)
    # error 1σ por bin asumiendo independencia
    err = np.sqrt(H2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return H, err, centers

def equivalent_events(w: np.ndarray) -> float:
    # N_equiv = (sum w)^2 / sum w^2
    s1 = np.sum(w)
    s2 = np.sum(w**2)
    return float((s1**2) / (s2 + 1e-300))