import numpy as np
import pygame
import sys
import InitializeScene as Init
import Minkowski as mink
import KerrSchild as KS
import BlackHoleOrbiter as BHO
import ParticleMergers as merge

    # -----------------------------------------------------------------------
    # MAIN - CLEAN BODY TRIAD VERSION (only this section is improved)
    # -----------------------------------------------------------------------

def Start():
    pygame.init()
    width, height = 720, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Science Engine v0.01")
    clock = pygame.time.Clock()

    particles, observer_name, dlambda = Init.run_example()
    ghosts = []
    frame_idx = 0
    observer = {p.name: p for p in particles}[observer_name]

    prev_lambdas = {}  # name → last λ_ret the observer saw

    # Body triad (world space)
    w_body = np.array([0,0,0],dtype=float)
    # Convention: X = right, Y = up, Z = forward  (matches Godot feel + Python depth = component 3)
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0])
    Z = np.array([0.0, 1.0, 0.0])

    R0 = np.eye(4)

    boost_strength = 0.06
    rot_strength   = 0.001

    return screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height





def Update(screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height):

    keys = pygame.key.get_pressed()

    # ---------- Body-frame input ----------
    boost_input = np.zeros(3)
    rot_input = np.zeros(3)

    w_world = X * w_body[0] + Y * w_body[1] + Z * w_body[2]

    # Pure spatial rotation (your preferred method)
    R = mink.lorentz_transform_matrix(0.0, 0.0, 0.0,
                                      w_world[0], w_world[1], w_world[2])

    # Rotate the triad (extract the spatial 3×3 block)
    R0 = R0 @ R
    R3 = np.linalg.inv(R[1:4, 1:4])

    # Movement (WASD + Space/Shift)  – body frame
    if keys[pygame.K_w]:          boost_input[1] += 1  # Forward
    if keys[pygame.K_s]:          boost_input[1] -= 1  # Backward
    if keys[pygame.K_a]:          boost_input[0] -= 1  # Left
    if keys[pygame.K_d]:          boost_input[0] += 1  # Right
    if keys[pygame.K_SPACE]:      boost_input[2] += 1  # Up
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        boost_input[2] -= 1  # Down

    # Rotation (Arrow keys + Q/E)  – body frame
    if keys[pygame.K_UP]:         rot_input[0] += 1  # Pitch up
    if keys[pygame.K_DOWN]:       rot_input[0] -= 1  # Pitch down
    if keys[pygame.K_LEFT]:       rot_input[2] += 1  # Yaw left
    if keys[pygame.K_RIGHT]:      rot_input[2] -= 1  # Yaw right
    if keys[pygame.K_q]:          rot_input[1] += 1  # Roll left
    if keys[pygame.K_e]:          rot_input[1] -= 1  # Roll right

    # ---------- Body-frame boost ----------
    if np.linalg.norm(boost_input) > 1e-6:
        body_dir = boost_input / np.linalg.norm(boost_input)
        world_dir = body_dir[0] * X + body_dir[1] * Y + body_dir[2] * Z
        Boost = mink.boost_matrix_from_3vel(world_dir * boost_strength)
        observer.vel = Boost @ observer.vel
        observer.spin = Boost @ observer.spin

    # ---------- Body-frame rotation using your lorentz_transform_matrix ----------
    if np.linalg.norm(rot_input) > 1e-6:
        # Body-frame generator components
        body_dir0 = rot_input / np.linalg.norm(rot_input)
        # world_dir0 = body_dir0[0]*X + body_dir0[1]*Y + body_dir0[2]*Z
        w_body += body_dir0 * rot_strength

    # R3 =
    X = R3 @ X
    Y = R3 @ Y
    Z = R3 @ Z

    # Generate the spin from the observer's angular velocity
    observer.spin = np.array((0.0, w_world[0], w_world[1], w_world[2]))

    # Final constraints
    observer.vel = mink.normalize_4velocity(observer.vel)
    observer.spin = KS.Initialize_spin_vector(observer.spin, observer.vel)

    # ---------- Physics (unchanged) ----------
    view_events, t_obs, particles, ghosts = BHO.run_nbody_step(
        particles, ghosts, observer_name, dlambda, frame_idx
    )

    # Apply mergers (returns both live list and updated ghosts)
    particles, ghosts = merge.apply_mergers(particles, ghosts)

    observer = merge.rebind_observer(observer, particles)
    observer_name = observer.name

    ghosts = merge.prune_ghosts(ghosts, observer)

    # ---------- Render ----------
    screen.fill((0, 0, 0))

    for ev in view_events:
        name = ev['name']
        lam = ev['lambda_src_ret']

        # Δλ since the last time this source was seen by the observer
        if name in prev_lambdas:
            delta_lambda = lam - prev_lambdas[name]
        else:
            delta_lambda = 0.0  # first appearance

        # Store for next frame
        prev_lambdas[name] = lam

        # Safety: ignore negative or tiny values (can happen with ghosts / clamping)
        if delta_lambda < 1e-8:
            delta_lambda = 1e-8

        # ----- brightness model -----
        # Simple linear scaling with a soft clamp.
        # You can later replace this with any function of Δλ.
        brightness = min(1.0, delta_lambda * 0.5 / dlambda)  # tune the 3.0 factor
        # brightness is now in [0.15 … 1.0]

        grey = int(255 * brightness)
        color = (grey, grey, grey)
        size_factor = 1

        x4 = R0 @ ev["x4_obs"]
        Xv, Yv, Zv = x4[1], x4[2], x4[3]

        if Zv > 0.1:
            sx = int(width / 2 + 540 * Xv / Zv)
            sy = int(height / 2 - 540 * Yv / Zv)

            if 0 <= sx < width and 0 <= sy < height:
                dist = max(np.sqrt(Xv * Xv + Yv * Yv + Zv * Zv), 1.0)
                radius = max(1, int(540 * ev['src_mass'] * size_factor / dist))
                pygame.draw.circle(screen, color, (sx, sy), radius)

    # Clean up prev_lambdas for particles that are no longer visible
    # (optional but keeps the dict from growing forever)
    visible_names = {ev['name'] for ev in view_events}
    prev_lambdas = {k: v for k, v in prev_lambdas.items() if k in visible_names}
    pygame.display.flip()
    clock.tick(60)
    frame_idx += 1

    return screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height

def play():
    screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height = Start()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height = Update(screen, clock, ghosts, frame_idx, observer, prev_lambdas, w_body, X, Y, Z, R0, boost_strength, rot_strength, particles, observer_name, dlambda, width, height)
    pygame.quit()
    sys.exit()


