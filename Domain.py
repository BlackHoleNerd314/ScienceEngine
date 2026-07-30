import numpy as np
import Orbiter as BHO
import Minkowski as mink
cutoff = 1/137.036
eps = 1e-7
def RelativityCheck(particle: BHO.Particle) -> None:
    v3 = mink.three_velocity_from_4velocity(particle.vel)
    if np.linalg.norm(v3) < cutoff:
        particle.IsRelativistic = False
    else:
        particle.IsRelativistic = True
def GravityCheck(particle: BHO.Particle,
                        sources: list) -> None:
    bound = 0
    for src in sources:
        reducedM = (src.mass*particle.mass)/(src.mass+particle.mass+eps)
        relV4 = src.vel - particle.vel
        v = np.linalg.norm(relV4[1:4])
        KE = reducedM*(v**2)/2 + eps
        PE = src.mass / (np.linalg.norm(src.pos - particle.pos) + eps)
        bound += (PE / KE)
    if bound < cutoff:
        particle.IsGravitational = False
    else:
        particle.IsGravitational = True
def solve_relativistic(particles, observer_name, dlambda):
    view_events = []
    observer = {p.name: p for p in particles}[observer_name]
    observer.pos = [0,0,0,0]
    observer.vel = [1,0,0,0]
    for p in particles:
        if p is observer:
            continue
        while p.pos[0] < dlambda:
            x4 = p.pos
            x = x4[1]
            y = x4[2]
            z = x4[3]
            p.vel = mink.normalize_4velocity(p.vel)
            v4 = p.vel
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            t = x4[0]
            t = t - r
            x = x + dlambda * v4[1]
            y = y + dlambda * v4[2]
            z = z + dlambda * v4[3]
            t = t + dlambda * v4[0]
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            t = t + r
            p.pos = [t, x, y, z]
            p.proper_time += dlambda
            p.record()
        view_events.append({
            "name": p.name,
            "x4_obs": p.pos.copy(),
            "v4_obs": p.vel.copy(),
            'j4_obs': p.spin.copy(),
            'lambda_src_ret': p.proper_time,
            'src_mass': p.mass
        })
    return view_events
def solve_classical(particles, observer_name, dlambda):
    view_events = []
    observer = {p.name: p for p in particles}[observer_name]
    observer.pos = [0,0,0,0]
    observer.vel = [1,0,0,0]
    for p in particles:
        if p is observer:
            continue
        p.pos[1:4] += (p.vel[1:4]) * dlambda
        p.pos[0]   += dlambda
        p.proper_time += dlambda
        p.vel[0] = 1
        p.record()
        view_events.append({
            'name': p.name,
            'x4_obs': p.pos.copy(),
            'v4_obs': p.vel.copy(),
            'j4_obs': p.spin.copy(),
            'lambda_src_ret': p.proper_time,
            'src_mass': p.mass
        })
    return view_events
def solve_gravity(particles, observer_name, dlambda):
    view_events = []
    observer = {p.name: p for p in particles}[observer_name]
    observer.pos = np.array([0, 0, 0, 0], dtype=float)
    observer.vel = np.array([1, 0, 0, 0], dtype=float)
    dt = dlambda
    for pi in particles:
        pi.vel[0] = 1
        ai = np.zeros(3)
        for pj in particles:
            if pi == pj:
                continue
            dx = pj.pos[1:4] - pi.pos[1:4]
            r2 = float(np.dot(dx, dx))
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)
            ai += (pj.mass / r2) * dx / r
        pi.vel[1:4] += ai * dt
        pi.pos[1:4] += (pi.vel[1:4]) * dt
        pi.pos[0] += dt
        pi.proper_time += dt
    for p in particles:
        p.pos -= observer.pos
        p.vel -= observer.vel
        p.vel[0] = 1
        p.record()
        view_events.append({
            'name': p.name,
            'x4_obs': p.pos.copy(),
            'v4_obs': p.vel.copy(),
            'j4_obs': p.spin.copy(),
            'lambda_src_ret': p.proper_time,
            'src_mass': p.mass
        })
    return view_events
def Dispatcher(particles: list,
                   observer_name: str,
                   dlambda: float):
    for p in particles:
        RelativityCheck(p)
        GravityCheck(p,particles)
    for p in particles:
        if p.IsRelativistic and p.IsGravitational:
            return BHO.Rebased_Step(particles,
            observer_name,
            dlambda)
        if (not p.IsRelativistic) and p.IsGravitational:
            return solve_gravity(particles,
            observer_name,
            dlambda)
        if p.IsRelativistic and (not p.IsGravitational):
            return solve_relativistic(particles,
            observer_name,
            dlambda)
        if (not p.IsRelativistic) and (not p.IsGravitational):
            return solve_classical(particles,
            observer_name,
            dlambda)