import math
import numpy as np
import Minkowski as mink

# -----------------------------------------------------------------------
# Change 5: Kerr metric from Lorentz-covariant tetrad, no active transform
#
# The tetrad (from CGH.py) with all Lorentz parameters = 0:
#
#   e[μ, a] = δ[μ, a]  +  (H/2) L^μ  l_a
#
# where  l_a = η_{aν} L^ν  and  L^μ is the Kerr-Schild null vector.
# Spin axis is always Z in the source rest frame.
# This is a purely coordinate description — no passive boosts or rotations
# are applied, which would not correspond to any physical transformation.
# -----------------------------------------------------------------------

def _ks_r(a: float, x: float, y: float, z: float) -> float:
    """Kerr radial coordinate r solving  r^4 - (R²-a²)r² - a²z² = 0."""
    R2 = x*x + y*y + z*z
    Ra = R2 - a*a
    b3 = math.sqrt(Ra*Ra + 4.0*a*a*z*z)
    r2 = (Ra + b3) * 0.5
    return math.sqrt(max(r2, 0.0))


def _ks_H(m: float, a: float, r: float, z: float) -> float:
    """
    Kerr-Schild scalar  H = 2 m r³ / (r⁴ + a² z²).
    Factor of 2 is absorbed here (matches CGH.py convention) so that
    g_μν = η_μν + H L_μ L_ν  (rather than  η_μν + 2V λ_μ λ_ν).
    """
    denom = r**4 + a*a*z*z
    if denom < 1e-30:
        return 0.0
    return 2.0 * m * r**3 / denom


def _ks_L(a: float, r: float, x: float, y: float, z: float) -> np.ndarray:
    """
    Kerr-Schild null vector  L^μ  (contravariant), spin along +Z.
    Null with respect to both η and g.
    """
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
    """

    The tetrad  e[μ,a] = I[μ,a] + (H/2) L^μ (η_{aν} L^ν)
    exactly as in CGH.py but with Lorentz parameters all zero (no active
    boost or rotation away from the standard Kerr-Schild chart):

    """
    x, y, z = float(x3[0]), float(x3[1]), float(x3[2])
    eta = mink.metric()
    r = _ks_r(a, x, y, z)
    H = _ks_H(m, a, r, z)
    L = _ks_L(a, r, x, y, z)         # L^μ  (upper)
    l = eta @ L                        # L_μ  (lower) = η_{μν} L^ν
    # Tetrad with Λ = I:  e[μ, a] = δ[μ,a] + (H/2) L[μ] l[a]
    e = np.eye(4) + 0.5 * H * np.outer(l, L)
    return e

def kerr_metric_restframe(m: float, a0: float, x3) -> np.ndarray:
    """
        Kerr metric g_μν at spatial position x3 = (x, y, z) in the source
        rest frame with spin axis along +Z.
                g[μ,ν] = η[a,b] e[μ,a] e[ν,b]
               = η[μ,ν] + H L[μ] L[ν]    (exploiting L·L = 0 w.r.t. η)

    The metric is stationary so only the spatial components of x3 matter.
    """
    # g[μ,ν] = η[a,b] e[μ,a] e[ν,b]
    eta = mink.metric()
    e = kerr_tetrad_restframe(m, a0, x3)
    return np.einsum('ab,ua,vb->uv', eta, e, e)

def christoffel_fd(x3, m: float, a: float, eps: float = 1e-6) -> np.ndarray:
    """
    Christoffel symbols  Γ^μ_{αβ}  via centred finite differences.
    The metric is stationary, so only spatial derivatives are non-zero.
    Returns shape (4, 4, 4):  Gamma[mu, alpha, beta].
    """
    g0 = kerr_metric_restframe(m, a, x3)
    ginv = np.linalg.inv(g0)

    # ∂_i g_{μν}  for i ∈ {x, y, z}  →  dg[spatial_index, μ, ν]
    dg_spatial = np.zeros((3, 4, 4))
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        gp = kerr_metric_restframe(m, a, np.array(x3) + dx)
        gm = kerr_metric_restframe(m, a, np.array(x3) - dx)
        dg_spatial[i] = (gp - gm) / (2.0 * eps)

    # Full  ∂_α g_{μν}:  α=0 (time) is zero for stationary metric
    dg = np.zeros((4, 4, 4))          # dg[α, μ, ν]
    dg[1] = dg_spatial[0]
    dg[2] = dg_spatial[1]
    dg[3] = dg_spatial[2]

    # Γ^μ_{αβ} = ½ g^{μν} (∂_α g_{νβ} + ∂_β g_{να} − ∂_ν g_{αβ})
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
    """
    Geodesic 4-acceleration:  du^μ/dλ = −Γ^μ_{αβ} u^α u^β.
    """
    return -np.einsum('mab,a,b->m', Gamma, u, u)

def geodesic_spin_transport(j: np.ndarray, u: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """
    Geodesic 4-spin parallel transport:  dj^μ/dλ = −Γ^μ_{αβ} j^α u^β.
    """
    return -np.einsum('mab,a,b->m', Gamma, j, u)

def Initialize_spin_vector(j: np.ndarray, u:np.ndarray):
    """
    Applies the physical constraint at t=0, such that
    j_μ u^μ = 0
    And it will evolve from there.
    """
    j[0] = (j[1]*u[1]+j[2]*u[2]+j[3]*u[3])/u[0]
    return j

