import Particle as P
import numpy as np
import Minkowski as mink
import math
import KerrSchild as KS

# -----------------------------------------------------------------------
# N-body acceleration using retarded cache + frame-boosted geodesics
# -----------------------------------------------------------------------

def compute_acceleration(particle: P.Particle,
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
        v3_src = mink.three_velocity_from_4velocity(u_ret)

        # 2. Boost: lab -> source rest frame ---------------------------------
        Lambda_to_src   = mink.boost_matrix_from_3vel(-v3_src)
        Lambda_from_src = mink.lorentz_inverse(Lambda_to_src)

        u_src  = Lambda_to_src @ u_ret
        dx_src = Lambda_to_src @ (particle.pos - x_ret)
        j_src = Lambda_to_src @ j_ret

        # 3. Decompose spin_J into magnitude (spin parameter a) + unit direction
        norm_j_src = mink.normalize_4spin(j_src)
        spin_a = math.sqrt(max(mink.interval(j_src), 1e-14))
        if spin_a > 1e-14:
            spin_unit = (norm_j_src[1], norm_j_src[2], norm_j_src[3])
        else:
            spin_unit = np.array([0.0, 0.0, 1.0])


        # 4. Active rotation: source rest frame -> Z-aligned frame -----------
        # rotation_to_align_spin returns the matrix that takes spin_unit -> Z.
        # Both the position offset and the 4-velocity must be rotated so
        # that the Kerr geometry (whose symmetry axis is Z in the tetrad)
        # is evaluated in the correct orientation.

        R_to_z, R_from_z = mink.rotation_to_align_spin(spin_unit)

        dx_rot = R_to_z @ dx_src
        u_rot  = R_to_z @ u_src
        j_rot  = R_to_z @ j_src
        x_rel  = dx_rot[1:4]          # spatial offset in Z-aligned frame

        # 5. Kerr geodesic in Z-aligned source frame (tetrad, untouched) -
        Gamma  = KS.christoffel_fd(x_rel, src.mass, spin_a)
        du_rot = KS.geodesic_acceleration(u_rot, Gamma)
        dj_rot = KS.geodesic_spin_transport(j_rot,u_rot,Gamma)

        # 6. Un-rotate: Z-aligned -> source rest frame -----------------------
        du_src_frame = R_from_z @ du_rot
        dj_src_frame = R_from_z @ dj_rot

        # 7. Un-boost: source rest frame -> lab frame ------------------------
        du_total += Lambda_from_src @ du_src_frame
        dj_total += Lambda_from_src @ dj_src_frame

    return du_total, dj_total


def Quantum(particle: P.Particle, dlambda: float):
    M = particle.mass
    v4 = particle.vel
    v3 = mink.three_velocity_from_4velocity(v4)
    dr = np.sqrt(dlambda/M)
    deltaX = np.zeros((4))
    deltaX[0] = 0
    deltaX[1] = np.random.normal() * dr
    deltaX[2] = np.random.normal() * dr
    deltaX[3] = np.random.normal() * dr
    Lambda_from_rest = mink.boost_matrix_from_3vel(v3)
    dX4 = Lambda_from_rest @ deltaX
    return dX4


# -----------------------------------------------------------------------
# Change 2: proper-time integration step
# -----------------------------------------------------------------------

def euler_step(particle: P.Particle, dlambda: float,
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

def advance_to_coord_time(particle: P.Particle, t_target: float,
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
def build_observer_view(observer: P.Particle,
                        particles: list,
                        ghosts: list = None):
    """
    Build a first-person view for `observer` at its current event.

    For each source:
      - Find the retarded event on the source worldline as seen by the observer.
      - Form the separation 4-vector in the lab frame.
      - Boost into the observer's instantaneous rest frame.
      - Rotate so that the observer's spin points along +Z.
    Returns a list of dicts, one per visible source.
    """
    """
        Build a first-person view for `observer` at its current event.

        Includes both live particles and ghosts.  A ghost is only visible
        while the retarded time the observer is seeing is still earlier
        than that ghost's death_time.  This keeps the pre-merger particles
        on screen until the light from the merger reaches the camera.

        All returned data are already in the observer's instantaneous
        rest frame (boosted + spin-aligned).  No lab-frame quantities
        are exposed to the renderer.
    """
    if ghosts is None:
        ghosts = []

    # Observer state in lab frame
    x_obs = observer.pos.copy()
    u_obs = observer.vel.copy()
    j_obs = observer.spin.copy()

    # 3-velocity of observer
    v3_obs = mink.three_velocity_from_4velocity(u_obs)

    # Boost lab -> observer rest frame
    Lambda_to_obs   = mink.boost_matrix_from_3vel(-v3_obs)
    Lambda_from_obs = mink.lorentz_inverse(Lambda_to_obs)

    # Observer spin direction in its rest frame (for orientation)
    j_obs_rest = Lambda_to_obs @ j_obs
    j_obs_rest = mink.normalize_4spin(j_obs_rest)
    spin_unit_obs = np.array([j_obs_rest[1], j_obs_rest[2], j_obs_rest[3]])

    # Rotate so that observer spin points along +Z
    R_to_z, R_from_z = mink.rotation_to_align_spin(spin_unit_obs)

    events = []

    # Candidates = live particles + ghosts
    candidates = list(particles) + list(ghosts)

    for src in candidates:
        if src is observer:
            continue

        # Retarded event of source as seen from observer's current position
        x_ret, u_ret, j_ret, lam_ret = src.retarded_position(x_obs)

        t_ret = float(x_ret[0])
        # ---------- visibility window ----------
        # Birth: particle is invisible until the light from its birth event
        # actually reaches the observer.  A clamped t_ret at the start of
        # the cache means the light has not arrived yet.
        if src.birth_time is not None:
            if t_ret < src.birth_time + 1e-8:
                continue
            # Still clamped to the very first cache entry → light has not arrived
            if abs(t_ret - src.cache_t_min) < 1e-8:
                continue

        # Death (ghosts): disappear once retarded time reaches death event
        # or the cache has ended
        if src.death_time is not None:
            if t_ret >= src.death_time - 1e-6:
                continue
            if t_ret >= src.cache_t_max - 1e-6:
                continue
        # ---------------------------------------

        # Separation in lab frame: source_ret - observer_now
        dx_lab = x_ret - x_obs

        # Boost into observer rest frame
        dx_obs = Lambda_to_obs @ dx_lab

        # Rotate so that observer spin is +Z
        dx_obs_z = R_to_z @ dx_obs

        events.append({
            'name': src.name,
            'x4_obs': dx_obs,      # 4-position of source event in observer Z-aligned frame
            'lambda_src_ret': lam_ret,
            'src_mass': src.mass,
            'is_ghost': src.is_ghost,
            'death_time': src.death_time,
            't_ret': t_ret,
        })

    return events

def run_nbody_step(particles: list,
                   ghosts: list,
                   observer_name: str,
                   dlambda: float,
                   frame_idx: int,
                   print_every: int = 10):
    """
    Performs ONE observer-driven N-body frame step.

    Returns
    -------
    view_events : list of retarded, boosted, rotated source events
                  (already player-centric; includes visible ghosts)
    t_obs       : observer coordinate time
    particles   : (possibly updated) live particle list
    ghosts      : updated ghost list
    """

    particles_by_name = {p.name: p for p in particles}
    observer = particles_by_name[observer_name]

    # 1. Advance the observer one proper-time step
    other_particles = [p for p in particles if p is not observer]
    euler_step(observer, dlambda, other_particles)

    # 2. Target coordinate time defined by the observer
    t_target = observer.coord_time

    # 3. Advance all other live particles to the same coordinate time
    for p in other_particles:
        p_sources = [q for q in particles if q is not p]
        advance_to_coord_time(p, t_target, dlambda, p_sources)

    # 4. Build the player-centric view (live + still-visible ghosts)
    view_events = build_observer_view(observer, particles, ghosts)

    # 5. Print observer-view data (player-centric only)
    return view_events, t_target, particles, ghosts
