import Orbiter as BHO
import Minkowski as mink
import numpy as np
import Mergers as ab
import Domain
def run_example():
    M = 10
    r0 = 100
    v_c = np.sqrt(M / r0)
    p1 = BHO.Particle(
        pos4=np.array([-(1000.0-r0), 0.0, 1000.0-r0, 0.0]),
        vel4=mink.four_velocity_from_3velocity((v_c / 2, 0.0, 0.0)),
        mass=M,
        spin4=np.array([0.0, 0.5, 0.0, 0.0]),
        name="p1",
    )
    p2 = BHO.Particle(
        pos4=np.array([-(1000.0+r0), 0.0, 1000.0+r0, 0.0]),
        vel4=mink.four_velocity_from_3velocity((-v_c / 2, 0.0, 0.0)),
        mass=M,
        spin4=np.array([0.0, -0.4, 0.0, 0.3]),
        name="p2",
    )
    p0 = BHO.Particle(
        pos4=np.array([0.0, 0.0, 0.0, 0.0]),
        vel4=mink.four_velocity_from_3velocity((0.0, 0.0, 0.0)),
        mass=0.01,
        spin4=np.array([0.0, 0.0, 0.0, 0.01]),
        name="p0",
    )
    particles = [p1, p2, p0]
    observer_name = "p0"
    dlambda = 1
    return particles, observer_name, dlambda
def main():
    particles, observer_name, dlambda = run_example()
    observer = {p.name: p for p in particles}[observer_name]
    for frame_idx in range(0,1000):
        particles, view_events = Domain.Dispatcher(particles, observer_name, dlambda)
        particles = ab.apply_mergers(particles)
        observer = ab.rebind_observer(observer, particles)
        observer_name = observer.name
        for ev in view_events:
            x4 = ev["x4_obs"]
            print(x4[1:4])
if __name__ == "__main__":
    main()
