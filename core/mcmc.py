import numpy as np
from typing import Callable, Tuple

def metropolis(logp: Callable[[float], float], x0: float, prop_std: float, n: int, rng=np.random) -> Tuple[np.ndarray, float]:
    chain = np.empty(n); chain[0] = x0; acc = 0
    for i in range(1, n):
        cur = chain[i-1]
        prop = cur + rng.normal(0.0, prop_std)
        if np.log(rng.random()) < (logp(prop) - logp(cur)):
            chain[i] = prop; acc += 1
        else:
            chain[i] = cur
    return chain, acc / (n - 1)

def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x)
    x = x - np.mean(x)
    c = np.correlate(x, x, mode='full')[len(x)-1:]
    c = c / c[0]
    return c[:max_lag+1]

# ---- Ising 2D (Metropolis, campo nulo) ----
def ising_metropolis(L: int, beta: float, n_sweeps: int, rng=np.random):
    # spins iniciales ±1
    S = rng.choice([-1, 1], size=(L, L))
    # simple integer sampler compatible with legacy and Generator RNGs
    def randint(high):
        return int(rng.random() * high)
    # vecinos con contorno periódico
    def nn_sum(i, j):
        return (S[(i+1)%L, j] + S[(i-1)%L, j] + S[i, (j+1)%L] + S[i, (j-1)%L])
    E_hist, M_hist = [], []
    for _ in range(n_sweeps):
        for _ in range(L*L):
            i = randint(L); j = randint(L)
            dE = 2 * S[i, j] * nn_sum(i, j)  # ΔE al voltear el spin (J=1)
            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                S[i, j] *= -1
        # observables por barrido
        E = 0.0
        for i in range(L):
            for j in range(L):
                E -= S[i, j] * (S[(i+1)%L, j] + S[i, (j+1)%L])  # J=1, enlaces únicos
        M = np.sum(S)
        E_hist.append(E / (L*L))
        M_hist.append(M / (L*L))
    return S, np.array(E_hist), np.array(M_hist)
