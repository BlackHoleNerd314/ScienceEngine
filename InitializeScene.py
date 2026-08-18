import numpy as np
import Minkowski as mink
import Particle as P
import math

def spawn_grid(n, spacing):
    pts = []
    counter = 0

    for i in range(n):
        x = np.random.normal()*spacing
        y = np.random.normal()*spacing
        z = np.random.normal()*spacing

        pos4 = np.array([0.0, x, y, z])
        vel4 = mink.four_velocity_from_3velocity((0,0,0))

        spin = np.array([
            0.0,
            np.random.normal(),
            np.random.normal(),
            np.random.normal()
        ])

        pts.append(
            P.Particle(
                pos4=pos4,
                vel4=vel4,
                spin4=spin,
                mass=np.random.uniform(),
                name=f"p{counter}"
            )
        )
        counter += 1

    return pts

def run_example_debug(n,r):
    particles = []
    # add grid particles
    particles += spawn_grid(n, r)
    observer = particles[0]  # or whichever
    observer_name = observer.name
    dlambda = 1
    return particles, observer_name, dlambda

def run_example():
    M = 10
    r0 = 100
    v_c = math.sqrt(M / r0)

    p1 = P.Particle(
        pos4=np.array([0.0, 0.0, -r0, 0.0]),
        vel4=mink.four_velocity_from_3velocity((v_c / 2, 0.0, 0.0)),
        mass=M,
        spin4=np.array([0.0, 0.5, 0.0, 0.0]),
        name="p1",
    )
    p1.birth_time = p1.pos[0]
    p2 = P.Particle(
        pos4=np.array([0.0, 0.0, r0, 0.0]),
        vel4=mink.four_velocity_from_3velocity((-v_c / 2, 0.0, 0.0)),
        mass=M,
        spin4=np.array([0.0, -0.4, 0.0, 0.3]),
        name="p2",
    )
    p2.birth_time = p2.pos[0]
    p3 = P.Particle(
        pos4=np.array([0.0, 0, 0, -r0*15]),
        vel4=mink.four_velocity_from_3velocity((0.0, 0.0, -0.1)),
        mass=M,
        spin4=np.array([0.0, 0.0, 0.0, 1.0]),
        name="p3",
    )
    p3.birth_time = p3.pos[0]

    particles = [p1, p2, p3]
    observer_name = "p3"
    dlambda = 10

    return particles, observer_name, dlambda
