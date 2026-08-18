
# Here are the massive bargmann wigner equations
# using the real majorana gamma matrices
# for a space-time algebra in the curved spacetime of
# the kerr geometry in cartesian kerr-schild coordinates
# with rotating black hole mass M and spin J equal
# to the particle mass m and spin s respectively,
# with M being a free parameter alongside the coordinates t,x,y,z,
# for a wavefunction of a given fixed spin of s*hbar,
# guiding a single relativistic bohmian trajectory,
# using a globally lorentz covariant tetrad
# e^a_%mu = delta^a_%mu + (H/2)L^aL_%mu, is chosen,
# raised and lowered by the minkowski metric eta, with signature -+++.

#Kerr spacetime geometry derivation:
#https://arxiv.org/pdf/0706.0622

#Spin connection derivation:
#https://ar5iv.labs.arxiv.org/html/2107.01114#A1
#https://arxiv.org/pdf/2107.01114

#Tetrad derivation:
#https://ar5iv.labs.arxiv.org/html/2203.16252
#https://arxiv.org/pdf/2203.16252

#Kerr-Schild coordinates derivation:
#http://www.fen.bilkent.edu.tr/~gurses/lorentzcovariant.pdf





import numpy as np
from scipy.linalg import expm


T0 = np.array([[0,0,0,-1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]])
X0 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
Y0 = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])
Z0 = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,-1,0]])

gamma = np.zeros((4,4,4))
gamma[0,:,:] = T0
gamma[1,:,:] = X0
gamma[2,:,:] = Y0
gamma[3,:,:] = Z0

def LorentzTransform(Kx,Ky,Kz,Jx,Jy,Jz):
 # Define general Lorentz Transformation matrix
 M = np.array([[0,Kx,Ky,Kz],
 [Kx,0,-Jz,Jy],
 [Ky,Jz,0,-Jx],
 [Kz,-Jy,Jx,0]])
 # Compute Matrix Exponential of Generating Matrix
 Lorentz = expm(M)
 return Lorentz

def MinkowskiMetric():
    # The spacetime invariant interval
    N = np.zeros((4,4))
    N[0,0] = -1
    N[1,1] = 1
    N[2,2] = 1
    N[3,3] = 1
    return N

def radius(a,x,y,z):
    R2 = x*x + y*y + z*z
    Ra = R2 - a*a
    AZ = a*a*z*z
    b3 = np.sqrt(Ra*Ra + 4*AZ)
    r2 = (Ra + b3)/2
    r = np.sqrt(r2)
    return r

def Potential(m,a,r,z):
    AZ = a * a * z * z
    H1 = 2 * m * r * r * r
    H2 = r * r * r * r + AZ
    H = H1 / H2
    return H

def Vector(a,r,x,y,z):
    L = np.zeros(4)
    L[0] = 1
    L[1] = (r * x + a * y) / (r * r + a * a)
    L[2] = (r * y - a * x) / (r * r + a * a)
    L[3] = z / r
    return L
def Tetrad(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz):
    X0 = np.array([t, x, y, z])
    Λ = LorentzTransform(Kx, Ky, Kz, Jx, Jy, Jz)  # Λ^μ{}_a
    X = np.linalg.solve(Λ, X0)                    # back to Kerr–Schild Cartesian

    eta = MinkowskiMetric()
    x0, y0, z0 = X[1], X[2], X[3]

    r = radius(a, x0, y0, z0)
    H = Potential(m, a, r, z0)
    L = Vector(a, r, x0, y0, z0)                  # L^μ

    # lower index: l_μ = η_{μν} L^ν
    l = eta @ L

    # transform to Lorentz frame index: L0_a = Λ^μ{}_a l_μ = (Λ.T @ l)_a
    L0 = Λ.T @ l

    # tetrad: e_μ{}^a = Λ_μ{}^a + (H/2) L_μ L0^a
    e0 = Λ.copy()
    e0 += 0.5 * H * np.outer(L, L0)

    return e0  # shape (4,4): e[μ,a]

def KerrMetric(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz):
    e0 = Tetrad(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)  # e[μ,a]
    eta = MinkowskiMetric()                                # η[a,b]
    g = np.einsum('ab,ua,vb->uv', eta, e0, e0)
    return g


diff0 = 1e-9

def Connection(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz):
    d = diff0
    dt = dx = dy = dz = d

    # metric at the point
    G = KerrMetric(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    g = np.linalg.inv(G)

    # metrics at shifted points
    Gt  = KerrMetric(m, a, t+dt, x,    y,    z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gt0 = KerrMetric(m, a, t-dt, x,    y,    z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gx  = KerrMetric(m, a, t,    x+dx, y,    z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gx0 = KerrMetric(m, a, t,    x-dx, y,    z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gy  = KerrMetric(m, a, t,    x,    y+dy, z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gy0 = KerrMetric(m, a, t,    x,    y-dy, z,    Kx, Ky, Kz, Jx, Jy, Jz)
    Gz  = KerrMetric(m, a, t,    x,    y,    z+dz, Kx, Ky, Kz, Jx, Jy, Jz)
    Gz0 = KerrMetric(m, a, t,    x,    y,    z-dz, Kx, Ky, Kz, Jx, Jy, Jz)

    # dG[μ,ν,ρ] = ∂_ρ g_{μν}
    dG = np.zeros((4, 4, 4))
    dG[:, :, 0] = (Gt - Gt0) / (2 * dt)
    dG[:, :, 1] = (Gx - Gx0) / (2 * dx)
    dG[:, :, 2] = (Gy - Gy0) / (2 * dy)
    dG[:, :, 3] = (Gz - Gz0) / (2 * dz)

    # B[μ,ν,σ] = ∂_μ g_{σν} + ∂_ν g_{σμ} - ∂_σ g_{μν}
    B = np.zeros((4, 4, 4))
    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                B[mu, nu, sigma] = (
                    dG[sigma, nu, mu] +
                    dG[sigma, mu, nu] -
                    dG[mu, nu, sigma]
                )

    # Γ^ρ_{μν} = 1/2 g^{ρσ} B[μ,ν,σ]
    Gamma = 0.5 * np.einsum('rs,mns->rmn', g, B)

    return Gamma  # shape (4,4,4): Γ[ρ,μ,ν]

def CurvedGammas(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz):
    e = Tetrad(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz)   # e[μ,a]
    # Gamma[μ,i,j] = e[μ,a] * gamma[a,i,j]
    Gamma = np.einsum('ma,aij->mij', e, gamma)
    return Gamma

def Mass_Term():
    Mass_gamma = np.eye(4)
    for dim0 in range(0,4):
        Mass_gamma = Mass_gamma @ gamma[dim0,:,:] # Majorana Cartesian Gammas
    return Mass_gamma

def Term(m,a,x,y,z):
    r = radius(a, x, y, z)
    H = Potential(m, a, r, z)
    L = Vector(a, r, x, y, z)
    # Construct Kerr Spacetime
    H0 = np.zeros((4, 4))
    for u in range(0, 4):
        for v in range(0, 4):
            H0[u, v] = H * L[v] * L[u] / 2
    return H0

def DerivateTerm(m, a, x, y, z):
    d = diff0
    dx = dy = dz = d

    hx  = Term(m, a,     x+dx, y,    z,    )
    hx0 = Term(m, a,     x-dx, y,    z,    )
    hy  = Term(m, a,     x,    y+dy, z,    )
    hy0 = Term(m, a,     x,    y-dy, z,    )
    hz  = Term(m, a,     x,    y,    z+dz, )
    hz0 = Term(m, a,     x,    y,    z-dz, )

    dG0 = np.zeros((4, 4, 4))
    dG0[:, :, 0] = 0
    dG0[:, :, 1] = (hx - hx0) / (2 * dx)
    dG0[:, :, 2] = (hy - hy0) / (2 * dy)
    dG0[:, :, 3] = (hz - hz0) / (2 * dz)
    return dG0

# Spin connection for Kerr-Schild tetrads (exact simplification per Alawadhi et al. 2107.01114 Appendix A1)
# (ω_μ)_ab = ∂_b e_aμ − ∂_a e_bμ   → only perturbation term contributes

def SpinConnection(m,a,x,y,z):
    h0 = DerivateTerm(m,a,x,y,z)
    Omega = np.zeros((4,4,4))
    for A in range(0,4):
        for B in range(0,4):
            for mu in range(0,4):
                Omega[A,B,mu] = (h0[A,mu,B] - h0[mu,B,A])/2
    return Omega


def Sigma(A,B):
    return (gamma[A,:,:] @ gamma[B,:,:] - gamma[B,:,:] @ gamma[A,:,:])/2


def SpinorConnection(m,a,x,y,z):
    omega = SpinConnection(m, a, x, y, z)
    Omega0 = np.zeros((4,4,4))
    for A in range(0,4):
        for B in range(0,4):
            for mu in range(0,4):
                Omega0[mu,:,:] += omega[A, B, mu] * Sigma(A, B)
    return Omega0

def DiracStep(M,x,y,z,Psi,Psi_x,Psi_y,Psi_z):
    m = M
    a = 1/(2*m)
    Gamma = CurvedGammas(m,a,0,x,y,z,0,0,0,0,0,0)
    omega = SpinorConnection(m, a, x, y, z)
    gamma_Tinv = np.linalg.inv(Gamma[0, :, :])
    Psi_t = - gamma_Tinv @ ( Mass_Term() @ (Psi * M)
    + Gamma[1, :, :] @ (Psi_x + omega[1,:,:] @ Psi)
    + Gamma[2, :, :] @ (Psi_y + omega[2,:,:] @ Psi)
    + Gamma[3, :, :] @ (Psi_z + omega[3,:,:] @ Psi)
    ) - omega[0,:,:] @ Psi
    #Psi += Psi_t
    return Psi_t

def PsiInit(N):
    Psi = np.zeros((2*N+1,2*N+1,2*N+1,4),dtype=float)
    for x in range(-N,N+1):
        for y in range(-N,N+1):
            for z in range(-N,N+1):
                Psi[x+N,y+N,z+N,:] = [0,0,0,0]
# WIP on my black hole-particle duality hypothesis
#print(DiracStep(13,3,4,12,np.array([0.1,0,0,0]),np.array([0,0,0.3,0]),np.array([0,0.2,0,0]),np.array([0,0,0,0.5])))


# Tests
def MinkowskiMetricTest(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz):
    e0 = np.linalg.inv(Tetrad(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz))
    g = KerrMetric(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz)
    h = np.zeros((4,4))
    for u in range(0,4):
        for v in range(0,4):
            for a0 in range(0,4):
                for b in range(0,4):
                    h[u,v] = h[u,v] + g[a0,b]*e0[u,a0]*e0[v,b]
    eta0 = MinkowskiMetric()
    print(h-eta0)
def KerrMetricTest(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz):
    X0 = np.array([t, x, y, z])
    delta0 = LorentzTransform(Kx, Ky, Kz, Jx, Jy, Jz)
    X = np.linalg.inv(delta0) @ X0
    x0 = X[1]
    y0 = X[2]
    z0 = X[3]
    N = MinkowskiMetric()
    r = radius(a,x0,y0,z0)
    H = Potential(m,a,r,z0)
    L = Vector(a,r,x0,y0,z0)
    # Construct Kerr Spacetime
    G = np.zeros((4,4))
    for u in range(0,4):
        for v in range(0,4):
            G[u,v] = N[u,v] + H*L[v]*L[u]
    G0 = KerrMetric(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz)
    print(G-G0)
def GammaFrameTest():
    eta = MinkowskiMetric()
    I = np.eye(4)
    for mu in range(0,4):
        for nu in range(0,4):
            Amat = gamma[mu,:,:]
            Bmat = gamma[nu,:,:]
            Cmat = np.matmul(Amat,Bmat)
            Cmat0 = np.matmul(Bmat,Amat)
            mat = (Cmat + Cmat0)/2
            Imat0 = I * eta[mu,nu]
            print(mat-Imat0)
#GammaFrameTest()
def VolumeTest(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz):
    e0 = Tetrad(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    G = KerrMetric(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    print(np.linalg.det(e0)-1)
    print(np.linalg.det(G)+1)
    print(np.linalg.det(T0)-1)
    print(np.linalg.det(X0)-1)
    print(np.linalg.det(Y0)-1)
    print(np.linalg.det(Z0)-1)
def LorentzTetradTest(m,a,t,x,y,z,Kx,Ky,Kz,Jx,Jy,Jz):
    G0 = (Tetrad(m,a,t,x,y,z,0,0,0,0,0,0))
    c0 = (LorentzTransform(Kx,Ky,Kz,Jx,Jy,Jz))
    X = c0 @ np.array([t,x,y,z])#@ np.linalg.inv(c0)
    cG0 = Tetrad(m,a,X[0],X[1],X[2],X[3],Kx,Ky,Kz,Jx,Jy,Jz)
    cG1 = G0 @ c0
    print(cG0-cG1)

def TestAll():
    m = np.random.rand()
    a = np.random.rand()
    t = np.random.rand()
    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()
    Kx = np.random.rand()
    Ky = np.random.rand()
    Kz = np.random.rand()
    Jx = np.random.rand()
    Jy = np.random.rand()
    Jz = np.random.rand()
    MinkowskiMetricTest(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    LorentzTetradTest(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    KerrMetricTest(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    VolumeTest(m, a, t, x, y, z, Kx, Ky, Kz, Jx, Jy, Jz)
    GammaFrameTest()








