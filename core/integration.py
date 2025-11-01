import numpy as np
from typing import Callable, Tuple

def mc_crude(g: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int, rng=np.random) -> Tuple[float, float]:
    u = rng.uniform(a, b, size=n)
    vals = g(u)
    I = (b - a) * np.mean(vals)
    # error estándar (1σ) del estimador
    s2 = np.var(vals, ddof=1)
    err = (b - a) * np.sqrt(s2 / n)
    return float(I), float(err)

def mc_stratified(g: Callable[[np.ndarray], np.ndarray], a: float, b: float, k: int, n_total: int, rng=np.random) -> Tuple[float, float]:
    # estratos iguales en longitud; asignación proporcional (n/k)
    edges = np.linspace(a, b, k + 1)
    n_per = n_total // k
    estimates, vars_ = [], []
    for j in range(k):
        aj, bj = edges[j], edges[j+1]
        u = rng.uniform(aj, bj, size=n_per)
        vals = g(u)
        m = np.mean(vals); s2 = np.var(vals, ddof=1)
        estimates.append((bj - aj) * m)
        vars_.append(((bj - aj) ** 2) * s2 / n_per)
    I = float(np.sum(estimates))
    err = float(np.sqrt(np.sum(vars_)))
    return I, err

def mc_importance(g: Callable[[np.ndarray], np.ndarray],
                  sampler_h,        # callable n -> x ~ h
                  pdf_h: Callable[[np.ndarray], np.ndarray],
                  n: int, rng=np.random) -> Tuple[float, float]:
    x = sampler_h(n)
    w = g(x) / (pdf_h(x) + 1e-300)
    I = float(np.mean(w))
    err = float(np.std(w, ddof=1) / np.sqrt(n))
    return I, err