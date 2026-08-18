import BlackHoleOrbiter as BHO
import Particle as P
import numpy as np
import math
import Minkowski as mink
import KerrSchild as KS

def rebind_observer(observer, particles):
    # Case 1: observer still exists
    if observer in particles:
        return observer

    # Case 2: observer was merged — find the particle that lists it as a parent
    for p in particles:
        if observer.name in p.parent_ids:
            return p

    # Case 3: fallback — choose nearest particle in spacetime
    return min(
        particles,
        key=lambda p: np.linalg.norm(p.pos4 - observer.pos4)
    )

def relative_ks_r(p_obs: P.Particle, p_src: P.Particle) -> float:
    """
    Kerr radial coordinate of p_obs as measured in the instantaneous
    rest frame of p_src after spin alignment to +Z.
    Exactly the same boost → rotate sandwich used by compute_acceleration.
    """
    # Use current states (pair is already synchronized and close)
    x_ret0, u_ret0, j_ret0, lam_ret0 = p_src.retarded_position(p_obs.pos.copy())
    v3_src = mink.three_velocity_from_4velocity(u_ret0)
    Lambda_to_src = mink.boost_matrix_from_3vel(-v3_src)

    dx_src = Lambda_to_src @ (p_obs.pos - x_ret0)

    j_src = Lambda_to_src @ j_ret0
    spin_a = math.sqrt(max(mink.interval(j_src), 1e-14))
    if spin_a > 1e-14:
        norm_j = mink.normalize_4spin(j_src)
        spin_unit = (norm_j[1], norm_j[2], norm_j[3])
    else:
        spin_unit = np.array([0.0, 0.0, 1.0])

    R_to_z, _ = mink.rotation_to_align_spin(spin_unit)
    dx_rot = R_to_z @ dx_src
    x_rel = dx_rot[1:4]

    return KS._ks_r(spin_a, x_rel[0], x_rel[1], x_rel[2])

def is_singularity(p1: P.Particle, p2: P.Particle, eps: float = 1) -> bool:
    """
    True when the relative Kerr radius of either particle inside the
    other has fallen below eps.  This is the point at which the
    finite-difference Christoffel symbols become unreliable.
    """
    r12 = relative_ks_r(p1, p2)
    r21 = relative_ks_r(p2, p1)
    return min(r12, r21) < eps

def renorm_via_tetrad(V_lab: np.ndarray,
                      src: P.Particle,
                      obs_pos: np.ndarray) -> np.ndarray:
    """
    Frame sandwich identical to compute_acceleration, but the final
    operation is a tetrad projection instead of a geodesic acceleration.

    lab  --boost-->  src rest  --rotate-->  Z-aligned
         <--un-boost--          <--un-rotate--

    Inside the Z-aligned chart the (copy of the) rest-frame tetrad
    converts the coordinate 4-vector into a local Lorentz (“flat”)
    4-vector.  The inverse sandwich then returns that local vector
    to the lab.  A transpose of the *copy* is used if needed; the
    original tetrad function is never modified.
    """
    v3_src = mink.three_velocity_from_4velocity(src.vel)
    Lambda_to_src   = mink.boost_matrix_from_3vel(-v3_src)
    Lambda_from_src = mink.lorentz_inverse(Lambda_to_src)

    V_src  = Lambda_to_src @ V_lab
    dx_src = Lambda_to_src @ (obs_pos - src.pos)

    j_src = Lambda_to_src @ src.spin
    spin_a = math.sqrt(max(mink.interval(j_src), 1e-14))
    if spin_a > 1e-14:
        norm_j = mink.normalize_4spin(j_src)
        spin_unit = (norm_j[1], norm_j[2], norm_j[3])
    else:
        spin_unit = np.array([0.0, 0.0, 1.0])

    R_to_z, R_from_z = mink.rotation_to_align_spin(spin_unit)

    V_rot  = R_to_z @ V_src
    dx_rot = R_to_z @ dx_src
    x_rel  = dx_rot[1:4]

    # --- tetrad in the compute frame (copy only) ---
    e = KS.kerr_tetrad_restframe(src.mass, spin_a, x_rel).copy()

    # Optional transpose of the *copy* (never of the original function).
    # Try e.T first; if mass explodes, switch to np.linalg.inv(e).
    # Both preserve the metric factors that encode time dilation.
    e_map = e          # or np.linalg.inv(e) if preferred after testing

    # Coordinate → local Lorentz
    V_local = e_map @ V_rot

    # --- return to lab ---
    V_src_frame = R_from_z @ V_local
    V_lab_ren   = Lambda_from_src @ V_src_frame
    return V_lab_ren

def merge_particles_tetrad(p1: P.Particle, p2: P.Particle) -> P.Particle:
    """
    Replace the naïve P1+P2 addition by the two-sided tetrad
    renormalization described in the request.

    1. Map (P1, J1) through p2’s geometry → contribution of parent 1
    2. Map (P2, J2) through p1’s geometry → contribution of parent 2
    3. Sum the renormalized vectors
    4. Reconstruct mass, 4-velocity and 4-spin from the sum
    """
    # Lab 4-momenta and 4-spins (identical to original merge_particles)
    P1 = p1.mass * p1.vel
    P2 = p2.mass * p2.vel
    J1 = p1.spin.copy()
    J2 = p2.spin.copy()

    # One-sided renormalizations
    P1_ren = renorm_via_tetrad(P1, p2, p1.pos)
    J1_ren = renorm_via_tetrad(J1, p2, p1.pos)

    P2_ren = renorm_via_tetrad(P2, p1, p2.pos)
    J2_ren = renorm_via_tetrad(J2, p1, p2.pos)

    # Sum (the tetrad factors have already suppressed the near-singularity blow-up)
    P_new = P1_ren + P2_ren
    J_new = J1_ren + J2_ren

    m_new = math.sqrt(max(-(mink.interval(P_new)), 1e-14))
    u_new = P_new / m_new

    # Position still the momentum-weighted average (or singularity location)
    x_new = (p1.mass * p1.pos + p2.mass * p2.pos) / (p1.mass + p2.mass)

    merged = P.Particle(
        pos4=x_new,
        vel4=u_new,
        spin4=J_new,
        mass=m_new,
        name=f"{p1.name}+{p2.name}"
    )
    merged.parent_ids = [p1.name, p2.name]
    p1.children.append(merged)
    p2.children.append(merged)
    return merged

def horizon_radius(p: P.Particle) -> float:
    M = 2*p.mass
    return M

def check_merger_condition(p1: P.Particle, p2: P.Particle) -> bool:
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

def apply_mergers(particles, ghosts, dlambda_plunge=1, max_plunge_steps=2000):
    """
    Keep original horizon test as the trigger that a pair has entered
    the strong-field regime.  Once triggered, integrate the pair in
    isolation until a singularity is reached, discarding any cache
    entries that were written after the singularity time.  Only then
    perform the tetrad-renormalized merger.
    """
    new_live = []
    new_ghosts = list(ghosts)
    skip = set()

    for i, p1 in enumerate(particles):
        if i in skip:
            continue
        for j, p2 in enumerate(particles):
            if j <= i or j in skip:
                continue

            # ----- original geometric horizon test (entry condition) -----
            if not check_merger_condition(p1, p2):
                continue

            # ----- plunge until singularity -----
            singularity_hit = False
            for _ in range(max_plunge_steps):
                if is_singularity(p1, p2):
                    singularity_hit = True
                    break

                # Guarded Euler steps (only the mutual interaction)
                try:
                    # record length before the step so we can discard
                    len1 = len(p1._cache)
                    len2 = len(p2._cache)

                    BHO.euler_step(p1, dlambda_plunge, [p2])
                    BHO.euler_step(p2, dlambda_plunge, [p1])

                    # crude NaN / explosion guard
                    if (np.linalg.norm(p1.pos[1:4]-p2.pos[1:4]) > (p1.mass + p2.mass) or
                        abs(p1.vel[0]) > 100 or abs(p2.vel[0]) > 100 or
                        not np.all(np.isfinite(p1.pos)) or not np.all(np.isfinite(p2.pos))):
                        # discard the bad entries
                        p1._cache = p1._cache[:len1]
                        p2._cache = p2._cache[:len2]
                        # restore last good state
                        if len1 > 0:
                            last = p1._cache[-1]
                            p1.pos, p1.vel, p1.spin = last['pos'], last['vel'], last['spin']
                            p1.proper_time = last['lam']
                        if len2 > 0:
                            last = p2._cache[-1]
                            p2.pos, p2.vel, p2.spin = last['pos'], last['vel'], last['spin']
                            p2.proper_time = last['lam']
                        singularity_hit = True
                        break
                except Exception:
                    # any numerical failure → treat as singularity
                    p1._cache = p1._cache[:len1]
                    p2._cache = p2._cache[:len2]
                    singularity_hit = True
                    break

            if not singularity_hit:
                # safety: force merge after max steps
                singularity_hit = True

            # ----- singularity reached: tetrad merger -----
            t_merge = max(p1.coord_time, p2.coord_time)

            merged = merge_particles_tetrad(p1, p2)

            merged.birth_time = t_merge

            p1.death_time = t_merge
            p2.death_time = t_merge
            p1.is_ghost = True
            p2.is_ghost = True

            new_ghosts.append(p1)
            new_ghosts.append(p2)
            new_live.append(merged)

            skip.add(i)
            skip.add(j)
            break
        else:
            if not p1.is_ghost:
                new_live.append(p1)

    return new_live, new_ghosts


def prune_ghosts(ghosts, observer):
    """Remove ghosts whose light has already reached the current observer."""
    alive_ghosts = []
    x_obs = observer.pos
    for g in ghosts:
        if g.death_time is None:
            alive_ghosts.append(g)
            continue
        # Quick check: if the latest possible retarded time is already past death
        if g.cache_t_max < g.death_time - 1e-6:
            # cache ended before death_time – keep for a moment longer if needed
            alive_ghosts.append(g)
            continue
        # Ask for the retarded time the observer would see right now
        x_ret, _, _, _ = g.retarded_position(x_obs)
        t_ret = float(x_ret[0])
        if t_ret < g.death_time - 1e-6 and t_ret < g.cache_t_max - 1e-6:
            alive_ghosts.append(g)
        # otherwise the ghost is no longer visible → drop it
    return alive_ghosts
