import numpy as np
import Minkowski
def _ks_r(a: float, x: float, y: float, z: float) -> float:
    R2 = x*x + y*y + z*z
    Ra = R2 - a*a
    b3 = np.sqrt(Ra*Ra + 4.0*a*a*z*z)
    r2 = (Ra + b3) * 0.5
    return np.sqrt(max(r2, 0.0))
def _ks_H(m: float, a: float, r: float, z: float) -> float:
    denom = r**4 + a*a*z*z
    if denom < 1e-30:
        return 0.0
    return 2.0 * m * r**3 / denom
def _ks_L(a: float, r: float, x: float, y: float, z: float) -> np.ndarray:
    L = np.zeros(4)
    denom = r*r + a*a
    L[0] = 1.0
    if denom > 1e-30:
        L[1] = (r*x + a*y) / denom
        L[2] = (r*y - a*x) / denom
    if r > 1e-30:
        L[3] = z / r
    return L
def kerr_tetrad_restframe(m: float, a: float, x3) -> np.ndarray:
    x, y, z = float(x3[0]), float(x3[1]), float(x3[2])
    eta = Minkowski.metric()
    r = _ks_r(a, x, y, z)
    H = _ks_H(m, a, r, z)
    L = _ks_L(a, r, x, y, z)
    l = eta @ L
    e = np.eye(4) + 0.5 * H * np.outer(l, L)
    return e
def kerr_metric_restframe(m: float, a0: float, x3) -> np.ndarray:
    eta = Minkowski.metric()
    e = kerr_tetrad_restframe(m, a0, x3)
    return np.einsum('ab,ua,vb->uv', eta, e, e)
def christoffel_fd(x3, m: float, a: float, eps: float = 1e-6) -> np.ndarray:
    g0 = kerr_metric_restframe(m, a, x3)
    ginv = np.linalg.inv(g0)
    dg_spatial = np.zeros((3, 4, 4))
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        gp = kerr_metric_restframe(m, a, np.array(x3) + dx)
        gm = kerr_metric_restframe(m, a, np.array(x3) - dx)
        dg_spatial[i] = (gp - gm) / (2.0 * eps)
    dg = np.zeros((4, 4, 4))
    dg[1] = dg_spatial[0]
    dg[2] = dg_spatial[1]
    dg[3] = dg_spatial[2]
    Gamma = np.zeros((4, 4, 4))
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                s = 0.0
                for nu in range(4):
                    s += ginv[mu, nu] * (dg[alpha, nu, beta]
                                       + dg[beta,  nu, alpha]
                                       - dg[nu, alpha, beta])
                Gamma[mu, alpha, beta] = 0.5 * s
    return Gamma
def geodesic_acceleration(u: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return -np.einsum('mab,a,b->m', Gamma, u, u)
def geodesic_spin_transport(j: np.ndarray, u: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return -np.einsum('mab,a,b->m', Gamma, j, u)
def Initialize_spin_vector(j: np.ndarray, u:np.ndarray):
    j[0] = (j[1]*u[1]+j[2]*u[2]+j[3]*u[3])/u[0]
    return j