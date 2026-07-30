import numpy as np
from scipy.linalg import expm
def metric() -> np.ndarray:
    return np.diag([-1.0, 1.0, 1.0, 1.0])
def lorentz_transform_matrix(Kx: float, Ky: float, Kz: float,
                              Jx: float, Jy: float, Jz: float) -> np.ndarray:
    M = np.array([[0.0,  Kx,   Ky,   Kz],
                  [Kx,  0.0,  -Jz,   Jy],
                  [Ky,   Jz,  0.0,  -Jx],
                  [Kz,  -Jy,   Jx,  0.0]])
    return expm(M)
def lorentz_inverse(Lambda: np.ndarray) -> np.ndarray:
    eta = metric()
    return eta @ Lambda.T @ eta
def boost_matrix_from_3vel(v3) -> np.ndarray:
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)
    if vmag < 1e-14:
        return np.eye(4)
    vmag_c = min(vmag, 1.0 - 1e-10)
    rapidity = np.atanh(vmag_c)
    nx, ny, nz = vx/vmag, vy/vmag, vz/vmag
    return lorentz_transform_matrix(nx*rapidity, ny*rapidity, nz*rapidity,
                                    0.0, 0.0, 0.0)
def rotation_to_align_spin(spin_unit) -> tuple:
    nx, ny, nz = float(spin_unit[0]), float(spin_unit[1]), float(spin_unit[2])
    cos_theta = max(-1.0, min(1.0, nz))
    if cos_theta > 1.0 - 1e-12:
        return np.eye(4), np.eye(4)
    if cos_theta < -1.0 + 1e-12:
        R     = lorentz_transform_matrix(0, 0, 0, np.pi, 0, 0)
        R_inv = lorentz_transform_matrix(0, 0, 0, -np.pi, 0, 0)
        return R, R_inv
    theta     = np.acos(cos_theta)
    sin_theta = np.sqrt(max(1.0 - cos_theta*cos_theta, 0.0))
    omx =  theta * ny / sin_theta
    omy = -theta * nx / sin_theta
    omz = 0.0
    R     = lorentz_transform_matrix(0, 0, 0,  omx,  omy, omz)
    R_inv = lorentz_transform_matrix(0, 0, 0, -omx, -omy, omz)
    return R, R_inv
def interval(v: np.ndarray) -> float:
    return -v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
def normalize_4velocity(u: np.ndarray) -> np.ndarray:
    n2 = interval(u)
    if abs(n2) < 1e-14:
        return u
    return u / np.sqrt(max(-n2, 1e-14))
def normalize_4spin(j: np.ndarray) -> np.ndarray:
    n2 = interval(j)
    if abs(n2) < 1e-14:
        return j
    return j / np.sqrt(max(n2, 1e-14))
def four_velocity_from_3velocity(v3) -> np.ndarray:
    vx, vy, vz = float(v3[0]), float(v3[1]), float(v3[2])
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)
    vmag = min(vmag, 1.0 - 1e-10)
    gamma = 1.0 / np.sqrt(1.0 - vmag*vmag)
    return np.array([gamma, gamma*vx, gamma*vy, gamma*vz])
def three_velocity_from_4velocity(u: np.ndarray) -> np.ndarray:
    if abs(u[0]) < 1e-14:
        return np.zeros(3)
    return u[1:4] / u[0]