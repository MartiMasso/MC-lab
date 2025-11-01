import numpy as np

def _rand_unit_vec(n, rng=np.random):
    # direcciones isotrópicas
    cos_th = rng.uniform(-1.0, 1.0, size=n)
    phi    = rng.uniform(0.0, 2*np.pi, size=n)
    sin_th = np.sqrt(1.0 - cos_th**2)
    x = sin_th * np.cos(phi)
    y = sin_th * np.sin(phi)
    z = cos_th
    return np.stack([x, y, z], axis=1)

def rambo_massless(n_particles: int, W: float, rng=np.random):
    # 1) energías exponenciales y direcciones isotrópicas
    e = -np.log(rng.uniform(size=n_particles))  # exp(1) distrib.
    n_hat = _rand_unit_vec(n_particles, rng)
    p = e[:, None] * n_hat  # momenta 3D provisionales
    q0 = e
    q  = p

    # 2) boost a reposo del total
    Q0 = np.sum(q0)
    Q  = np.sum(q, axis=0)
    beta = -Q / (Q0 + 1e-300)
    b2 = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - b2)

    # boost
    bp = (q @ beta)
    p_par = (bp[:, None] * beta[None, :]) / (b2 + 1e-300)
    p_perp = q - p_par
    q0p = gamma * (q0 + bp)
    qp  = p_perp + gamma * (p_par + beta[None, :] * q0[:, None])

    # 3) rescale energías para que sumen W
    scale = W / (np.sum(q0p) + 1e-300)
    E = q0p * scale
    P = qp * scale
    # salida 4-vector: (E, px, py, pz)
    return np.concatenate([E[:, None], P], axis=1)