import math
import numpy as np
from scipy.linalg import expm


# -----------------------------------------------------------------------
# Minkowski metric  η_μν = diag(-1,+1,+1,+1)
# -----------------------------------------------------------------------

def metric() -> np.ndarray:
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


def lorentz_inverse(Lambda: np.ndarray) -> np.ndarray:
    """
    Inverse of a Lorentz matrix:  Λ^{-1} = η Λ^T η.
    Works for any element of O(1,3).
    """
    eta = metric()
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

