import Orbiter as BHO
import numpy as np
import Minkowski as mink
def rebind_observer(observer, particles):
    if observer in particles:
        return observer
    for p in particles:
        if observer.name in p.parent_ids:
            return p
    return min(
        particles,
        key=lambda p: np.linalg.norm(p.pos4 - observer.pos4)
    )
def merge_particles(p1: BHO.Particle, p2: BHO.Particle) -> BHO.Particle:
    P1 = p1.mass * mink.normalize_4velocity(p1.vel)
    P2 = p2.mass * mink.normalize_4velocity(p2.vel)
    P_new = P1 + P2
    m_new = np.sqrt(max(-(mink.interval(P_new)), 1e-14))
    u_new = P_new / m_new
    J_new = p1.spin + p2.spin
    x_new = (p1.mass * p1.pos + p2.mass * p2.pos) / (p1.mass + p2.mass)
    merged = BHO.Particle(
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
def check_merger_condition(p1: BHO.Particle, p2: BHO.Particle) -> bool:
    x1 = p1.pos[1:4]
    x2 = p2.pos[1:4]
    r12 = float(np.linalg.norm(x1 - x2))
    R1 = p1.mass
    R2 = p2.mass
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