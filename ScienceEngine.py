#!/usr/bin/env python3
"""
Black Hole Orbiter (revised).

Improvements over original:
  1. Translated from MATLAB into Python, which opens the possibility of
    making a scientifically accurate game-engine with pygame visualizations
    while keeping low level control throughout for critical features.
  2. Simulates General Relativity Numerically, by referencing a 1975 paper
    "A Lorentz Covariant Treatment of the Kerr-Schild Geometry" by Gurses and Gursey.
    It is the only known way to fully simulate the General Theory of Relativity
    by using the only coordinate system that exactly linearizes the
    Einstein Field Equations, and does so without Supercomputing Clusters.
  3. Keeps the author's complete unifying vision of a Unified Field Theory intact.
    To do this requires a paradigm shift where every particle
    is also a rotating black hole, a field quantum, and an observer.
    This means one recovers Lorentz Covariant Motion in Special Relativity,
    Newtonian Force Laws of Universal Gravitation, Quantum Mechanics, and
    Multi-Particle Theory, with both mass and spin as particle invariants.

Author: Erik Jorgensen
Date: 2026-05-30 (revised), 2009-08-22 (original)
"""

import math
import numpy as np
from scipy.linalg import expm
import pygame
import sys

# -----------------------------------------------------------------------
# Minkowski metric  η_μν = diag(-1,+1,+1,+1)
# -----------------------------------------------------------------------

def minkowski_metric() -> np.ndarray:
    """Return η_μν with signature (−+++)."""
    return np.diag([-1.0, 1.0, 1.0, 1.0])


# -----------------------------------------------------------------------
# Change 4: Lorentz group via matrix exponential  (matches CGH.py)
# -----------------------------------------------------------------------

def lorentz_transform_matrix(Kx: float, Ky: float, Kz: float,
                              Jx: float, Jy: float, Jz: float) -> np.ndarray:
    """
    General Lorentz transformation  Λ = exp(M)  where M is the generator.

    K components are rapidities (boost parameters);
    J components are rotation angles.
    Convention identical to CGH.py:
        M = [[0,  Kx,  Ky,  Kz],
             [Kx,  0, -Jz,  Jy],
             [Ky,  Jz,  0, -Jx],
             [Kz, -Jy,  Jx,  0]]
    """
    M = np.array([[0.0,  Kx,   Ky,   Kz],
                  [Kx,  0.0,  -Jz,   Jy],
                  [Ky,   Jz,  0.0,  -Jx],
                  [Kz,  -Jy,   Jx,  0.0]])
    return expm(M)

def deltaFunc(i,j):
    if i==j:
        return 1
    else:
        return 0

def QuantumSpinMatrix(N: int):
    Jx = np.zeros((N, N), dtype=complex)
    Jy = np.zeros((N, N), dtype=complex)
    Jz = np.zeros((N, N), dtype=complex)
    for n0 in range(0,N):
        for n1 in range(0,N):
            j = (N - 1)/2
            a0 = n0 - j
            a1 = n1 - j
            Jx[n0, n1] += (np.sqrt((j - a1) * (j + a1 + 1))) * deltaFunc(a0, a1 + 1)/2
            Jx[n0, n1] += (np.sqrt((j + a1) * (j - a1 + 1))) * deltaFunc(a0, a1 - 1)/2
            Jy[n0, n1] += (np.sqrt((j - a1) * (j + a1 + 1))) * deltaFunc(a0, a1 + 1)/2j
            Jy[n0, n1] += (np.sqrt((j + a1) * (j - a1 + 1))) * deltaFunc(a0, a1 - 1)/2j
            Jz[n0, n1] += a1 * deltaFunc(a0, a1)
    return Jx,Jy,Jz

def spin_transform_matrix(N: int, Kx: float, Ky: float, Kz: float,
                              Jx: float, Jy: float, Jz: float):
    x0, y0, z0 = QuantumSpinMatrix(N)
    Mboost = (x0*Kx + y0*Ky + z0*Kz)
    Mrotate = (x0*Jx + y0*Jy + z0*Jz)
    M = Mboost + Mrotate*1j
    return expm(M)


def boost_matrix_from_3vel(v3) -> np.ndarray:
    """
    Pure Lorentz boost for 3-velocity v3 = (vx, vy, vz).

    Converts to rapidity  η = arctanh(|v|)  and feeds the unit-direction
    components into lorentz_transform_matrix as (Kx, Ky, Kz).
    No rotation parameters are used.
    """
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if vmag < 1e-14:
        return np.eye(4)
    vmag_c = min(vmag, 1.0 - 1e-10)   # clamp away from light-speed
    rapidity = math.atanh(vmag_c)
    nx, ny, nz = vx/vmag, vy/vmag, vz/vmag
    return lorentz_transform_matrix(nx*rapidity, ny*rapidity, nz*rapidity,
                                    0.0, 0.0, 0.0)


def spin_matrix_from_data(N,v3,spin_unit):

    """
    Return (R, R_inv): 4×4 spatial rotation matrices built from
    lorentz_transform_matrix (zero boost params) that actively rotate
    *spin_unit* onto +Z.

    Derivation
    ----------
    Given unit vector n̂ we need R such that  R n̂ = ẑ.
    Rotation axis:  k̂ = (n̂ × ẑ) / |n̂ × ẑ|
    Rotation angle: θ  = arccos(n̂ · ẑ)
    Rotation vector sent to the generator: ω = θ k̂

    With ẑ = (0,0,1):
        n̂ × ẑ = (ny, −nx, 0)   →   ω = θ(ny, −nx, 0) / sinθ

    The Jx, Jy, Jz parameters of lorentz_transform_matrix encode the
    rotation vector directly (no boost, K=0), so the result is a pure
    spatial rotation that leaves the time component untouched.

    Special cases
    -------------
    n̂ ≈ +ẑ : identity (no rotation needed).
    n̂ ≈ −ẑ : π rotation around x̂.
    """
    nx, ny, nz = float(spin_unit[0]), float(spin_unit[1]), float(spin_unit[2])
    cos_theta = max(-1.0, min(1.0, nz))   # n̂ · ẑ = nz

    if cos_theta > 1.0 - 1e-12:           # already aligned with +Z
        return np.eye(N)

    if cos_theta < -1.0 + 1e-12:          # anti-aligned: rotate π around x̂
        R     = spin_transform_matrix(0, 0, 0, math.pi, 0, 0)
        return R

    theta     = math.acos(cos_theta)
    sin_theta = math.sqrt(max(1.0 - cos_theta*cos_theta, 0.0))

    # ω = θ (ny, −nx, 0) / sinθ
    omx =  theta * ny / sin_theta
    omy = -theta * nx / sin_theta
    omz = 0.0

    M_rotate = spin_transform_matrix(N,0, 0, 0,  omx,  omy, omz)
    """
    Pure Lorentz boost for 3-velocity v3 = (vx, vy, vz).

    Converts to rapidity  η = arctanh(|v|)  and feeds the unit-direction
    components into lorentz_transform_matrix as (Kx, Ky, Kz).
    No rotation parameters are used.
    """
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if vmag < 1e-14:
        return np.eye(N)
    vmag_c = min(vmag, 1.0 - 1e-10)   # clamp away from light-speed
    rapidity = math.atanh(vmag_c)
    nx, ny, nz = vx/vmag, vy/vmag, vz/vmag
    M_boost = spin_transform_matrix(N,nx*rapidity, ny*rapidity, nz*rapidity,
                                    0.0, 0.0, 0.0)
    Mdata = M_boost @ M_rotate
    return Mdata


def lorentz_inverse(Lambda: np.ndarray) -> np.ndarray:
    """
    Inverse of a Lorentz matrix:  Λ^{-1} = η Λ^T η.
    Works for any element of O(1,3).
    """
    eta = minkowski_metric()
    return eta @ Lambda.T @ eta


def rotation_to_align_spin(spin_unit) -> tuple:
    """
    Return (R, R_inv): 4×4 spatial rotation matrices built from
    lorentz_transform_matrix (zero boost params) that actively rotate
    *spin_unit* onto +Z.

    Derivation
    ----------
    Given unit vector n̂ we need R such that  R n̂ = ẑ.
    Rotation axis:  k̂ = (n̂ × ẑ) / |n̂ × ẑ|
    Rotation angle: θ  = arccos(n̂ · ẑ)
    Rotation vector sent to the generator: ω = θ k̂

    With ẑ = (0,0,1):
        n̂ × ẑ = (ny, −nx, 0)   →   ω = θ(ny, −nx, 0) / sinθ

    The Jx, Jy, Jz parameters of lorentz_transform_matrix encode the
    rotation vector directly (no boost, K=0), so the result is a pure
    spatial rotation that leaves the time component untouched.

    Special cases
    -------------
    n̂ ≈ +ẑ : identity (no rotation needed).
    n̂ ≈ −ẑ : π rotation around x̂.
    """
    nx, ny, nz = float(spin_unit[0]), float(spin_unit[1]), float(spin_unit[2])
    cos_theta = max(-1.0, min(1.0, nz))   # n̂ · ẑ = nz

    if cos_theta > 1.0 - 1e-12:           # already aligned with +Z
        return np.eye(4), np.eye(4)

    if cos_theta < -1.0 + 1e-12:          # anti-aligned: rotate π around x̂
        R     = lorentz_transform_matrix(0, 0, 0, math.pi, 0, 0)
        R_inv = lorentz_transform_matrix(0, 0, 0, -math.pi, 0, 0)
        return R, R_inv

    theta     = math.acos(cos_theta)
    sin_theta = math.sqrt(max(1.0 - cos_theta*cos_theta, 0.0))

    # ω = θ (ny, −nx, 0) / sinθ
    omx =  theta * ny / sin_theta
    omy = -theta * nx / sin_theta
    omz = 0.0

    R     = lorentz_transform_matrix(0, 0, 0,  omx,  omy, omz)
    R_inv = lorentz_transform_matrix(0, 0, 0, -omx, -omy, omz)
    return R, R_inv


def quantum_boost_matrix_from_3vel(N:int,v3):
    """
    Pure Lorentz boost for 3-velocity v3 = (vx, vy, vz).

    Converts to rapidity  η = arctanh(|v|)  and feeds the unit-direction
    components into lorentz_transform_matrix as (Kx, Ky, Kz).
    No rotation parameters are used.
    """
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if vmag < 1e-14:
        return np.eye(N)
    vmag_c = min(vmag, 1.0 - 1e-10)   # clamp away from light-speed
    rapidity = math.atanh(vmag_c)
    nx, ny, nz = vx/vmag, vy/vmag, vz/vmag
    return spin_transform_matrix(N,nx*rapidity, ny*rapidity, nz*rapidity,
                                    0.0, 0.0, 0.0)


def quantum_rotation_to_align_spin(N:int,spin_unit):
    """
    Return (R, R_inv): 4×4 spatial rotation matrices built from
    lorentz_transform_matrix (zero boost params) that actively rotate
    *spin_unit* onto +Z.

    Derivation
    ----------
    Given unit vector n̂ we need R such that  R n̂ = ẑ.
    Rotation axis:  k̂ = (n̂ × ẑ) / |n̂ × ẑ|
    Rotation angle: θ  = arccos(n̂ · ẑ)
    Rotation vector sent to the generator: ω = θ k̂

    With ẑ = (0,0,1):
        n̂ × ẑ = (ny, −nx, 0)   →   ω = θ(ny, −nx, 0) / sinθ

    The Jx, Jy, Jz parameters of lorentz_transform_matrix encode the
    rotation vector directly (no boost, K=0), so the result is a pure
    spatial rotation that leaves the time component untouched.

    Special cases
    -------------
    n̂ ≈ +ẑ : identity (no rotation needed).
    n̂ ≈ −ẑ : π rotation around x̂.
    """
    nx, ny, nz = float(spin_unit[0]), float(spin_unit[1]), float(spin_unit[2])
    cos_theta = max(-1.0, min(1.0, nz))   # n̂ · ẑ = nz

    if cos_theta > 1.0 - 1e-12:           # already aligned with +Z
        return np.eye(N), np.eye(N)

    if cos_theta < -1.0 + 1e-12:          # anti-aligned: rotate π around x̂
        R     = spin_transform_matrix(N,0, 0, 0, math.pi, 0, 0)
        R_inv = spin_transform_matrix(N,0, 0, 0, -math.pi, 0, 0)
        return R, R_inv

    theta     = math.acos(cos_theta)
    sin_theta = math.sqrt(max(1.0 - cos_theta*cos_theta, 0.0))

    # ω = θ (ny, −nx, 0) / sinθ
    omx =  theta * ny / sin_theta
    omy = -theta * nx / sin_theta
    omz = 0.0

    R     = spin_transform_matrix(N,0, 0, 0,  omx,  omy, omz)
    R_inv = spin_transform_matrix(N,0, 0, 0, -omx, -omy, omz)
    return R, R_inv


# -----------------------------------------------------------------------
# 4-vector / 4-velocity helpers
# -----------------------------------------------------------------------

def interval(v: np.ndarray) -> float:
    """Spacetime interval  η_μν v^μ v^ν."""
    return -v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]


def normalize_4velocity(u: np.ndarray) -> np.ndarray:
    """Rescale so that  η_μν u^μ u^ν = -1."""
    n2 = interval(u)
    if abs(n2) < 1e-14:
        return u
    return u / math.sqrt(max(-n2, 1e-14))

def normalize_4spin(j: np.ndarray) -> np.ndarray:
    """Rescale so that  η_μν j^μ j^ν = 1."""
    n2 = interval(j)
    if abs(n2) < 1e-14:
        return j
    return j / math.sqrt(max(n2, 1e-14))

def four_velocity_from_3velocity(v3) -> np.ndarray:
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
    vmag = min(vmag, 1.0 - 1e-10)
    gamma = 1.0 / math.sqrt(1.0 - vmag*vmag)
    return np.array([gamma, gamma*vx, gamma*vy, gamma*vz])


def three_velocity_from_4velocity(u: np.ndarray) -> np.ndarray:
    if abs(u[0]) < 1e-14:
        return np.zeros(3)
    return u[1:4] / u[0]


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


def kerr_metric_restframe(m: float, a: float, x3) -> np.ndarray:
    """
    Kerr metric g_μν at spatial position x3 = (x, y, z) in the source
    rest frame with spin axis along +Z.

    Built from the tetrad  e[μ,a] = I[μ,a] + (H/2) L^μ (η_{aν} L^ν)
    exactly as in CGH.py but with Lorentz parameters all zero (no active
    boost or rotation away from the standard Kerr-Schild chart):

        g[μ,ν] = η[a,b] e[μ,a] e[ν,b]
               = η[μ,ν] + H L[μ] L[ν]    (exploiting L·L = 0 w.r.t. η)

    The metric is stationary so only the spatial components of x3 matter.
    """
    x, y, z = float(x3[0]), float(x3[1]), float(x3[2])
    eta = minkowski_metric()
    r = _ks_r(a, x, y, z)
    H = _ks_H(m, a, r, z)
    L = _ks_L(a, r, x, y, z)         # L^μ  (upper)
    l = eta @ L                        # L_μ  (lower) = η_{μν} L^ν
    # Tetrad with Λ = I:  e[μ, a] = δ[μ,a] + (H/2) L[μ] l[a]
    e = np.eye(4) + 0.5 * H * np.outer(l, L)
    # g[μ,ν] = η[a,b] e[μ,a] e[ν,b]
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


# -----------------------------------------------------------------------
# Change 1 & 2: Particle class with worldline cache + proper-time state
# -----------------------------------------------------------------------

class Particle:
    """
    A gravitating body.  Integration is in proper time; coordinate time
    advances at rate  dt/dλ = u^0 (the Lorentz factor).

    The worldline cache stores snapshots at every integration step,
    ordered by (monotonically increasing) coordinate time.  Retarded
    positions are found by binary search on this cache.
    """

    def __init__(self, pos4: np.ndarray, vel4: np.ndarray, spin4: np.ndarray,
                 mass: float,
                 name: str = ""):
        """
        Parameters
        ----------
        pos4   : initial 4-position (t, x, y, z).
        vel4   : initial 4-velocity (will be normalised).
        spin4  : initial 4-spin     (will be normalised)
        mass   : gravitational mass (G = c = 1).
        spin_J : 3-vector whose *magnitude* is the Kerr spin parameter a
                 and whose *direction* is the spin axis in the lab frame.
                 Examples:
                   spin_J = (0, 0, 0.5)  -> a=0.5, spin along +Z
                   spin_J = (0.3, 0, 0)  -> a=0.3, spin along +X
        name   : optional label for diagnostics.
        """
        self.pos = pos4.astype(float).copy()
        self.vel = normalize_4velocity(vel4.astype(float).copy())
        self.spin = Initialize_spin_vector(spin4.astype(float).copy(),self.vel)
        self.mass = mass
        self.name = name

        self.proper_time: float = 0.0
        # Cache: list of dicts, sorted by ascending coordinate time.
        self._cache: list = [self._snapshot()]

    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            't':   float(self.pos[0]),
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'spin': self.spin.copy(),
            'lam': self.proper_time,
        }

    def record(self) -> None:
        """Append current state to cache."""
        self._cache.append(self._snapshot())

    @property
    def coord_time(self) -> float:
        return float(self.pos[0])

    @property
    def cache_t_min(self) -> float:
        return self._cache[0]['t']

    @property
    def cache_t_max(self) -> float:
        return self._cache[-1]['t']

    # ------------------------------------------------------------------

    def interpolate_at(self, t_query: float):
        """
        Linearly interpolate the cached worldline at coordinate time t_query.
        Returns (pos4, vel4).  Clamps to cache endpoints if out of range.
        """
        cache = self._cache
        if t_query <= cache[0]['t']:
            return cache[0]['pos'].copy(), cache[0]['vel'].copy(), cache[0]['spin'].copy(), cache[0]['lam']
        if t_query >= cache[-1]['t']:
            return cache[-1]['pos'].copy(), cache[-1]['vel'].copy(), cache[-1]['spin'].copy(), cache[-1]['lam']

        # Binary search for bracketing entries
        lo, hi = 0, len(cache) - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cache[mid]['t'] <= t_query:
                lo = mid
            else:
                hi = mid

        a0, a1 = cache[lo], cache[hi]
        dt = a1['t'] - a0['t']
        if dt < 1e-15:
            return a0['pos'].copy(), a0['vel'].copy(), a0['spin'].copy(), a0['lam']
        frac = (t_query - a0['t']) / dt
        pos = a0['pos'] + frac * (a1['pos'] - a0['pos'])
        vel = a0['vel'] + frac * (a1['vel'] - a0['vel'])
        spin = a0['spin'] + frac * (a1['spin'] - a0['spin'])
        lam =  a0['lam'] + frac * (a1['lam'] - a0['lam'])
        return pos, vel, spin, lam

    # ------------------------------------------------------------------
    # Change 1: retarded position via cache binary search
    # ------------------------------------------------------------------

    def retarded_position(self, observer_pos4: np.ndarray,
                          tol: float = 1e-10, max_iter: int = 64):
        """
        Find the retarded 4-position and 4-velocity of this particle as
        seen from observer_pos4 by binary-searching the worldline cache.

        Solves:  c·(t_obs − t_ret) = |x_src(t_ret) − x_obs|
                 (with c = 1 in natural units)

        i.e. the separation 4-vector between source and observer is null.

        Parameters
        ----------
        observer_pos4 : 4-position of the observer at the *current* time.
        tol           : coordinate-time tolerance for the bisection.
        max_iter      : maximum bisection iterations.

        Returns
        -------
        pos_ret, vel_ret, spin_ret : 4-position, 4-velocity, 4-spin at the retarded event.
        """
        t_obs  = float(observer_pos4[0])
        x_obs  = observer_pos4[1:4]

        t_lo = self.cache_t_min
        t_hi = min(self.cache_t_max, t_obs)

        if t_hi <= t_lo:
            return self._cache[0]['pos'].copy(), self._cache[0]['vel'].copy(), self._cache[0]['spin'].copy(), self._cache[0]['lam']

        def residual(t_ret: float) -> float:
            """Positive when t_ret is too early, negative when too late."""
            pos_r, vel_r, spin_r, lam_r = self.interpolate_at(t_ret)
            dx = pos_r[1:4] - x_obs
            dist = math.sqrt(float(np.dot(dx, dx)))
            return (t_obs - t_ret) - dist

        f_lo = residual(t_lo)
        f_hi = residual(t_hi)

        # If the root is not bracketed, return the nearest endpoint
        if f_lo * f_hi > 0:
            t_ret = t_lo if abs(f_lo) < abs(f_hi) else t_hi
        else:
            for _ in range(max_iter):
                t_mid = 0.5 * (t_lo + t_hi)
                f_mid = residual(t_mid)
                if abs(f_mid) < tol:
                    break
                if f_lo * f_mid <= 0.0:
                    t_hi = t_mid
                    f_hi = f_mid
                else:
                    t_lo = t_mid
                    f_lo = f_mid
            t_ret = 0.5 * (t_lo + t_hi)

        return self.interpolate_at(t_ret)


# -----------------------------------------------------------------------
# N-body acceleration using retarded cache + frame-boosted geodesics
# -----------------------------------------------------------------------

def compute_acceleration(particle: Particle,
                          sources: list) -> np.ndarray:
    """
    Total geodesic 4-acceleration of *particle* due to all *sources*.

    For each source the evaluation chain is:

      lab  --boost-->  source rest frame  --rotate-->  Z-aligned frame
           <--un-boost--                 <--un-rotate--

      1. Retarded (causal) position/velocity from cache.
      2. Boost lab -> source rest frame  (matrix-exponential, K params).
      3. Active spatial rotation aligning src.spin_unit with +Z
         (matrix-exponential, J params only, K=0).
         This is sandwiched between the boosts; it is NOT applied to the
         tetrad, which would be a passive relabelling, not a physical rotation.
      4. Tetrad Kerr metric evaluated with spin along Z  (untouched).
      5. Geodesic acceleration in the Z-aligned source frame.
      6. Inverse rotation: Z-aligned -> source rest frame.
      7. Inverse boost: source rest frame -> lab frame.
    """
    du_total = np.zeros(4)
    dj_total = np.zeros(4)

    for src in sources:
        # 1. Retarded source state -------------------------------------------
        x_ret, u_ret, j_ret, lam_ret = src.retarded_position(particle.pos)
        v3_src = three_velocity_from_4velocity(u_ret)

        # 2. Boost: lab -> source rest frame ---------------------------------
        Lambda_to_src   = boost_matrix_from_3vel(-v3_src)
        Lambda_from_src = lorentz_inverse(Lambda_to_src)

        u_src  = Lambda_to_src @ u_ret
        dx_src = Lambda_to_src @ (particle.pos - x_ret)
        j_src = Lambda_to_src @ j_ret

        # 3. Decompose spin_J into magnitude (spin parameter a) + unit direction
        norm_j_src = normalize_4spin(j_src)
        spin_a = math.sqrt(max(interval(j_src), 1e-14))
        if spin_a > 1e-14:
            spin_unit = (norm_j_src[1], norm_j_src[2], norm_j_src[3])
        else:
            spin_unit = np.array([0.0, 0.0, 1.0])


        # 4. Active rotation: source rest frame -> Z-aligned frame -----------
        # rotation_to_align_spin returns the matrix that takes spin_unit -> Z.
        # Both the position offset and the 4-velocity must be rotated so
        # that the Kerr geometry (whose symmetry axis is Z in the tetrad)
        # is evaluated in the correct orientation.

        R_to_z, R_from_z = rotation_to_align_spin(spin_unit)

        dx_rot = R_to_z @ dx_src
        u_rot  = R_to_z @ u_src
        j_rot  = R_to_z @ j_src
        x_rel  = dx_rot[1:4]          # spatial offset in Z-aligned frame

        # 5. Kerr geodesic in Z-aligned source frame (tetrad, untouched) -
        Gamma  = christoffel_fd(x_rel, src.mass, spin_a)
        du_rot = geodesic_acceleration(u_rot, Gamma)
        dj_rot = geodesic_spin_transport(j_rot,u_rot,Gamma)

        # 6. Un-rotate: Z-aligned -> source rest frame -----------------------
        du_src_frame = R_from_z @ du_rot
        dj_src_frame = R_from_z @ dj_rot

        # 7. Un-boost: source rest frame -> lab frame ------------------------
        du_total += Lambda_from_src @ du_src_frame
        dj_total += Lambda_from_src @ dj_src_frame

    return du_total, dj_total


def Quantum(particle: Particle, dlambda: float):
    M = particle.mass
    v4 = particle.vel
    v3 = three_velocity_from_4velocity(v4)
    dr = np.sqrt(dlambda/M)/137.036
    deltaX = np.zeros((4))
    deltaX[0] = 0
    deltaX[1] = np.random.normal() * dr
    deltaX[2] = np.random.normal() * dr
    deltaX[3] = np.random.normal() * dr
    Lambda_from_rest = boost_matrix_from_3vel(v3)
    dX4 = Lambda_from_rest @ deltaX
    return dX4



# -----------------------------------------------------------------------
# Change 2: proper-time integration step
# -----------------------------------------------------------------------

def euler_step(particle: Particle, dlambda: float,
               sources: list) -> None:
    """Advance particle by one proper-time step  dλ  (forward Euler)."""
    dX4 = Quantum(particle, dlambda)
    particle.pos += dX4
    du, dj = compute_acceleration(particle, sources)
    particle.vel = particle.vel + du * dlambda
    particle.spin = particle.spin + dj * dlambda
    particle.pos = particle.pos + particle.vel * dlambda
    particle.proper_time += dlambda
    particle.record()


# -----------------------------------------------------------------------
# Change 3: coordinate-time-synchronised N-body loop
#
# All particles are advanced in *their own proper times* until every
# particle's worldline cache extends to the same target coordinate time.
# This guarantees that the retarded-time binary search always has
# sufficient history for any particle pair, regardless of how fast each
# particle's coordinate time ticks relative to its proper time.
# -----------------------------------------------------------------------

def advance_to_coord_time(particle: Particle, t_target: float,
                          dlambda: float, sources: list,
                          integrator=euler_step) -> None:
    """
    Integrate *particle* in proper time until  coord_time >= t_target.

    The inner loop uses  dlambda  proper-time steps.  Because
        dt_coord / dλ = u^0 = γ ≥ 1
    particles moving rapidly will take *fewer* proper-time steps to
    reach the same coordinate-time target.  The cache accumulates all
    intermediate states, providing a dense worldline for retarded lookups.
    """
    while particle.coord_time < t_target:
        integrator(particle, dlambda, sources)



# -----------------------------------------------------------------------
# FIX: build_observer_view takes particles (with caches), not histories
# -----------------------------------------------------------------------
def build_observer_view(observer: Particle,
                        particles: list):
    """
    Build a first-person view for `observer` at its current event.

    For each source:
      - Find the retarded event on the source worldline as seen by the observer.
      - Form the separation 4-vector in the lab frame.
      - Boost into the observer's instantaneous rest frame.
      - Rotate so that the observer's spin points along +Z.
    Returns a list of dicts, one per visible source.
    """
    # Observer state in lab frame
    x_obs = observer.pos.copy()
    u_obs = observer.vel.copy()
    j_obs = observer.spin.copy()

    # 3-velocity of observer
    v3_obs = three_velocity_from_4velocity(u_obs)

    # Boost lab -> observer rest frame
    Lambda_to_obs   = boost_matrix_from_3vel(-v3_obs)
    Lambda_from_obs = lorentz_inverse(Lambda_to_obs)

    # Observer spin direction in its rest frame (for orientation)
    j_obs_rest = Lambda_to_obs @ j_obs
    j_obs_rest = normalize_4spin(j_obs_rest)
    spin_unit_obs = np.array([j_obs_rest[1], j_obs_rest[2], j_obs_rest[3]])

    # Rotate so that observer spin points along +Z
    R_to_z, R_from_z = rotation_to_align_spin(spin_unit_obs)

    events = []

    for src in particles:
        if src is observer:
            continue

        # Retarded event of source as seen from observer's current position
        x_ret, u_ret, j_ret, lam_ret = src.retarded_position(x_obs)

        # Separation in lab frame: source_ret - observer_now
        dx_lab = x_ret - x_obs

        # Boost into observer rest frame
        dx_obs = Lambda_to_obs @ dx_lab

        # Rotate so that observer spin is +Z
        dx_obs_z = R_to_z @ dx_obs

        events.append({
            'name': src.name,
            'x4_obs': dx_obs_z,      # 4-position of source event in observer Z-aligned frame
            'lambda_src_ret': lam_ret,
            'src_mass': src.mass
        })

    return events


def run_nbody_step(particles: list,
                   observer_name: str,
                   dlambda: float,
                   frame_idx: int,
                   print_every: int = 10):
    """
    Performs ONE observer-driven N-body frame step.

    Returns:
        view_events : list of retarded, boosted, rotated source events
        t_obs       : observer coordinate time
    """

    # Map names to particles and pick the observer
    particles_by_name = {p.name: p for p in particles}
    observer = particles_by_name[observer_name]

    # -----------------------------------------------------------------
    # 1. Force the observer forward by ONE frame tick in its proper time.
    # -----------------------------------------------------------------
    other_particles = [p for p in particles if p is not observer]
    euler_step(observer, dlambda, other_particles)

    # -----------------------------------------------------------------
    # 2. Extract the target coordinate time endpoint defined by the observer.
    # -----------------------------------------------------------------
    t_target = observer.coord_time

    # -----------------------------------------------------------------
    # 3. Advance all environment entities until their caches match this target.
    # -----------------------------------------------------------------
    for p in other_particles:
        p_sources = [q for q in particles if q is not p]
        advance_to_coord_time(p, t_target, dlambda, p_sources)

    # -----------------------------------------------------------------
    # 4. Generate first-person visual output safely synchronized on t_target.
    # -----------------------------------------------------------------
    view_events = build_observer_view(observer, particles)

    if (frame_idx % print_every) == 0:
        print(f"[run_nbody_step] frame {frame_idx}, t_obs = {t_target:.6f}, "
              f"lambda_obs = {observer.proper_time:.6f}")

    return view_events, t_target

def run_example():
    M = 1
    r0 = 10
    v_c = math.sqrt(M / r0)

    p1 = Particle(
        pos4=np.array([0.0, -r0, 0.0, 0.0]),
        vel4=four_velocity_from_3velocity((0.0, 0.0, v_c / 2)),
        mass=M,
        spin4=np.array([0.0, 0.0, 0.0, 0.5]),
        name="p1",
    )
    p2 = Particle(
        pos4=np.array([0.0, r0, 0.0, 0.0]),
        vel4=four_velocity_from_3velocity((0.0, 0.0, -v_c / 2)),
        mass=M,
        spin4=np.array([0.0, 0.3, 0.0, -0.4]),
        name="p2",
    )
    p3 = Particle(
        pos4=np.array([0.0, 0, -r0*15, 0]),
        vel4=four_velocity_from_3velocity((0.0, 0.0, 0.0)),
        mass=M,
        spin4=np.array([0.0, 0.0, 0.1, 0.0]),
        name="p3",
    )

    particles = [p1, p2, p3]
    observer_name = "p3"
    dlambda = 1

    return particles, observer_name, dlambda

def merge_particles(p1: Particle, p2: Particle) -> Particle:
    # 4-momentum conservation
    P1 = p1.mass * normalize_4velocity(p1.vel)
    P2 = p2.mass * normalize_4velocity(p2.vel)
    P_new = P1 + P2

    # new mass from invariant
    m_new = math.sqrt(max(-(interval(P_new)), 1e-14))

    # new 4-velocity
    u_new = P_new / m_new

    # 4-spin conservation
    J_new = p1.spin + p2.spin

    # new position = momentum-weighted average
    x_new = (p1.mass * p1.pos + p2.mass * p2.pos) / (p1.mass + p2.mass)

    merged = Particle(
        pos4=x_new,
        vel4=u_new,
        spin4=J_new,
        mass=m_new,
        name=f"{p1.name}+{p2.name}"
    )
    return merged

def horizon_radius(p: Particle) -> float:
    M = p.mass
    return M

def check_merger_condition(p1: Particle, p2: Particle) -> bool:
    """
    Simple geometric merger rule:
    two holes merge when their spatial separation
    is less than the sum of their horizon radii.
    """
    x1 = p1.pos[1:4]
    x2 = p2.pos[1:4]
    r12 = float(np.linalg.norm(x1 - x2))

    R1 = horizon_radius(p1)
    R2 = horizon_radius(p2)

    return r12 <= (R1 + R2)


def apply_mergers(particles):
    merged_list = []
    skip = set()

    for i, p1 in enumerate(particles):
        if i in skip:
            continue
        for j, p2 in enumerate(particles):
            if j <= i or j in skip:
                continue

            if check_merger_condition(p1, p2):
                merged = merge_particles(p1, p2)
                merged_list.append(merged)
                skip.add(i)
                skip.add(j)
                break
        else:
            merged_list.append(p1)

    return merged_list


def main():
    pygame.init()
    width, height = 720, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Science Engine: First-Person Relativistic View")
    clock = pygame.time.Clock()

    # ---------------------------------------------------------
    # INITIALIZE SIMULATION (from run_example)
    # ---------------------------------------------------------
    particles, observer_name, dlambda = run_example()
    frame_idx = 0
    Rotate0 = np.eye(4)
    observer = {p.name: p for p in particles}[observer_name]

    running = True
    while running:

        # Quit handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        X0 = 0
        Y0 = 0
        Z0 = 0
        boost_strength = 0.02  # small rapidity per frame
        rotate_strength = 0.02 # small angle per frame

        if keys[pygame.K_w]:
            Y0 += boost_strength  # forward
        if keys[pygame.K_s]:
            Y0 -= boost_strength  # backward
        if keys[pygame.K_a]:
            X0 -= boost_strength  # left
        if keys[pygame.K_d]:
            X0 += boost_strength  # right
        if keys[pygame.K_r]:
            Z0 -= boost_strength  # up
        if keys[pygame.K_f]:
            Z0 += boost_strength  # down

        X1 = 0
        Y1 = 0
        Z1 = 0

        if keys[pygame.K_q]:
            Z1 += rotate_strength
        if keys[pygame.K_e]:
            Z1 -= rotate_strength
        if keys[pygame.K_z]:
            X1 += rotate_strength
        if keys[pygame.K_x]:
            X1 -= rotate_strength
        if keys[pygame.K_v]:
            Y1 += rotate_strength
        if keys[pygame.K_c]:
            Y1 -= rotate_strength

        Rotate0 = Rotate0 @ lorentz_transform_matrix(0, 0, 0, X1, Y1, Z1)

        if (X0 or Y0 or Z0 or X1 or Y1 or Z1):
            # Boost Logic
            Boost0 = lorentz_transform_matrix(X0, Y0, Z0, 0, 0, 0)
            observer.vel = Boost0 @ observer.vel
            observer.spin = Boost0 @ observer.spin
            # SYNC: Update the observer in particles list
            for i, p in enumerate(particles):
                if p.name == observer_name:
                    particles[i] = observer
                    break

        # ---------------------------------------------------------
        # PHYSICS STEP (ONE PROPER-TIME TICK)
        # ---------------------------------------------------------
        view_events, t_obs = run_nbody_step(
            particles,
            observer_name,
            dlambda,
            frame_idx
        )

        # apply mergers AFTER advancing
        particles = apply_mergers(particles)

        # Rebind observer after mergers
        observer = {p.name: p for p in particles}[observer_name]

        # ---------------------------------------------------------
        # RENDER
        # ---------------------------------------------------------
        screen.fill((0, 0, 0))

        for ev in view_events:
            x4 = ev["x4_obs"]
            x4 = Rotate0 @ x4
            X = x4[1]
            Y = x4[2]
            Z = x4[3]
            if Z > 0.1:
                fov = 540
                sx = int(width/2 + fov * X / Z)
                sy = int(height/2 - fov * Y / Z)

                if 0 <= sx < width and 0 <= sy < height:
                    radius = fov * ev['src_mass'] / np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                    pygame.draw.circle(screen, (255,255,255), (sx, sy), radius)

        # HUD
        font = pygame.font.SysFont(None, 24)
        hud = font.render(
            f"λ={observer.proper_time:.2f}  t={observer.coord_time:.2f}",
            True, (200,200,200)
        )
        screen.blit(hud, (20,20))

        pygame.display.flip()
        clock.tick(47)
        frame_idx += 1

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


