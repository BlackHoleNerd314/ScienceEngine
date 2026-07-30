import numpy as np
import Minkowski as mink
import KerrSchild as KS
class Particle:
    def __init__(self, pos4: np.ndarray, vel4: np.ndarray, spin4: np.ndarray,
                 mass: float,
                 name: str = ""):
        self.pos = pos4.astype(float).copy()
        self.vel = mink.normalize_4velocity(vel4.astype(float).copy())
        self.spin = KS.Initialize_spin_vector(spin4.astype(float).copy(),self.vel)
        self.mass = mass
        self.name = name
        self.IsRelativistic = True
        self.IsGravitational = True
        self.proper_time: float = 0.0
        self._cache: list = [self._snapshot()]
        self.parent_ids = []
        self.children = []
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
    def interpolate_at(self, t_query: float):
        cache = self._cache
        if t_query <= cache[0]['t']:
            return cache[0]['pos'].copy(), cache[0]['vel'].copy(), cache[0]['spin'].copy(), cache[0]['lam']
        if t_query >= cache[-1]['t']:
            return cache[-1]['pos'].copy(), cache[-1]['vel'].copy(), cache[-1]['spin'].copy(), cache[-1]['lam']
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
    def retarded_position(self, observer_pos4: np.ndarray,
                          tol: float = 1e-10, max_iter: int = 64):
        t_obs  = float(observer_pos4[0])
        x_obs  = observer_pos4[1:4]
        t_lo = self.cache_t_min
        t_hi = min(self.cache_t_max, t_obs)
        if t_hi <= t_lo:
            return self._cache[0]['pos'].copy(), self._cache[0]['vel'].copy(), self._cache[0]['spin'].copy(), self._cache[0]['lam']
        def residual(t_ret: float) -> float:
            pos_r, vel_r, spin_r, lam_r = self.interpolate_at(t_ret)
            dx = pos_r[1:4] - x_obs
            dist = np.sqrt(float(np.dot(dx, dx)))
            return (t_obs - t_ret) - dist
        f_lo = residual(t_lo)
        f_hi = residual(t_hi)
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
def compute_acceleration(particle: Particle,
                          sources: list) -> np.ndarray:
    du_total = np.zeros(4)
    dj_total = np.zeros(4)
    for src in sources:
        x_ret, u_ret, j_ret, lam_ret = src.retarded_position(particle.pos)
        v3_src = mink.three_velocity_from_4velocity(u_ret)
        Lambda_to_src   = mink.boost_matrix_from_3vel(-v3_src)
        Lambda_from_src = mink.lorentz_inverse(Lambda_to_src)
        u_src  = Lambda_to_src @ u_ret
        dx_src = Lambda_to_src @ (particle.pos - x_ret)
        j_src = Lambda_to_src @ j_ret
        norm_j_src = mink.normalize_4spin(j_src)
        spin_a = np.sqrt(max(mink.interval(j_src), 1e-14))
        if spin_a > 1e-14:
            spin_unit = (norm_j_src[1], norm_j_src[2], norm_j_src[3])
        else:
            spin_unit = np.array([0.0, 0.0, 1.0])
        R_to_z, R_from_z = mink.rotation_to_align_spin(spin_unit)
        dx_rot = R_to_z @ dx_src
        u_rot  = R_to_z @ u_src
        j_rot  = R_to_z @ j_src
        x_rel  = dx_rot[1:4]
        Gamma  = KS.christoffel_fd(x_rel, src.mass, spin_a)
        du_rot = KS.geodesic_acceleration(u_rot, Gamma)
        dj_rot = KS.geodesic_spin_transport(j_rot,u_rot,Gamma)
        du_src_frame = R_from_z @ du_rot
        dj_src_frame = R_from_z @ dj_rot
        du_total += Lambda_from_src @ du_src_frame
        dj_total += Lambda_from_src @ dj_src_frame
    return du_total, dj_total
def euler_step(particle: Particle, dlambda: float,
               sources: list) -> None:
    du, dj = compute_acceleration(particle, sources)
    particle.vel = particle.vel + du * dlambda
    particle.spin = particle.spin + dj * dlambda
    particle.pos = particle.pos + particle.vel * dlambda
    particle.proper_time += dlambda
    particle.record()
def advance_to_coord_time(particle: Particle, t_target: float,
                          dlambda: float, sources: list,
                          integrator=euler_step) -> None:
    while particle.coord_time < t_target:
        integrator(particle, dlambda, sources)
def build_observer_view(observer: Particle,
                        particles: list):
    x_obs = observer.pos.copy()
    u_obs = observer.vel.copy()
    j_obs = observer.spin.copy()
    v3_obs = mink.three_velocity_from_4velocity(u_obs)
    Lambda_to_obs   = mink.boost_matrix_from_3vel(-v3_obs)
    j_obs_rest = Lambda_to_obs @ j_obs
    j_obs_rest = mink.normalize_4spin(j_obs_rest)
    spin_unit_obs = np.array([j_obs_rest[1], j_obs_rest[2], j_obs_rest[3]])
    R_to_z, _ = mink.rotation_to_align_spin(spin_unit_obs)
    events = []
    for src in particles:
        if src is observer:
            continue
        x_ret, u_ret, j_ret, lam_ret = src.retarded_position(x_obs)
        dx_lab = x_ret - x_obs
        dx_obs = Lambda_to_obs @ dx_lab
        u_obs = Lambda_to_obs @ u_ret
        j_obs = Lambda_to_obs @ j_ret
        dx_obs_z = R_to_z @ dx_obs
        u_obs_z = R_to_z @ u_obs
        j_obs_z = R_to_z @ j_obs
        events.append({
            'name': src.name,
            'x4_obs': dx_obs_z,
            'v4_obs': u_obs_z,
            'j4_obs': j_obs_z,
            'lambda_src_ret': lam_ret,
            'src_mass': src.mass
        })
    return events
def run_nbody_step(particles: list,
                   observer_name: str,
                   dlambda: float):
    particles_by_name = {p.name: p for p in particles}
    observer = particles_by_name[observer_name]
    other_particles = [p for p in particles if p is not observer]
    euler_step(observer, dlambda, other_particles)
    t_target = observer.coord_time
    for p in other_particles:
        p_sources = [q for q in particles if q is not p]
        advance_to_coord_time(p, t_target, dlambda, p_sources)
    view_events = build_observer_view(observer, particles)
    return view_events, t_target
def Rebased_Step(particles: list,
                   observer_name: str,
                   dlambda: float):
    observer = {p.name: p for p in particles}[observer_name]
    view_events, t_obs = run_nbody_step(particles, observer.name, dlambda)
    other_particles = [p for p in particles if p is not observer]
    for p in other_particles:
        for ev in view_events:
            name = ev["name"]
            x4 = ev["x4_obs"]
            v4 = ev["v4_obs"]
            j4 = ev["j4_obs"]
            if p.name == name:
                p.pos4 = x4
                p.vel4 = v4
                p.spin4 = j4
    observer.pos4 = np.array([0.0, 0.0, 0.0, 0.0])
    observer.vel4 = mink.four_velocity_from_3velocity((0.0, 0.0, 0.0))
    observer.mass = 0.0
    observer.spin4 = np.array([0.0, 0.0, 0.0, 1e-7])
    observer.name = "Player"
    view_events = build_observer_view(observer, particles)
    return particles, view_events