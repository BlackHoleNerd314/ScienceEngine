# ScienceEngine

**A real-time Kerr–Schild N-body engine in Python.**

ScienceEngine treats every particle as a rotating Kerr black hole in Cartesian Kerr–Schild coordinates. Interactions are computed by retarded linear superposition of geodesic accelerations and spin transport — not by summing metrics, and not by evolving a 3-D spacetime grid. The result is a player-centric relativistic game engine that runs on a laptop.

This is the working proof-of-concept for the **black-hole–particle duality** developed in *On the Nature of Things*.

---

## What this is (and is not)

This engine stays inside the Kerr–Schild class and exploits the linearity of the Einstein equations in that class (Gürses & Gürsey, 1975). It is:

- exact for a single Kerr (or Schwarzschild) source,
- a retarded, causal multi-body construction obtained by superposing *accelerations* after an active Lorentz-frame sandwich,
- interactive on consumer hardware,
- first-person: the camera is an observer particle with its own 4-velocity and 4-spin.

It is **not** unrestricted numerical relativity (BSSN / generalized harmonic evolution of a free metric on a 3-D grid). Traditional NR is the right tool for generic strong-field waveforms. ScienceEngine is the right tool for a grid-free, inspectable, playable Kerr–Schild universe.

---

## 1. Staying inside the Kerr–Schild sector

### The metric class

A Kerr–Schild metric has the form

\[
g_{\mu\nu} = \eta_{\mu\nu} + H\, L_\mu L_\nu
\]

where \(\eta_{\mu\nu}\) is Minkowski, \(L^\mu\) is null with respect to both \(\eta\) and \(g\), and \(H\) is a scalar. In these coordinates the gravitational pseudo-energy-momentum tensor vanishes, so the Einstein equations become **linear** in \(H\). Gravity is not its own source. That is the 1975 result of Gürses and Gürsey.

Linearity is what makes a multi-body engine possible without a supercomputer. The price is that one must remain inside this class.

### Why we do not sum metrics

Adding two Kerr–Schild metrics generally produces a geometry that is no longer Kerr–Schild and no longer satisfies the linearized field equations. The principal null directions of the two sources would also be misaligned.

ScienceEngine never builds a total metric. For each source it evaluates the **exact analytic Kerr tetrad** of that source, computes the geodesic 4-acceleration and parallel transport of spin, then adds those 4-vectors. What is superposed is the *effect* of each source, which is the operation licensed by the linearity of the Einstein tensor in this coordinate system.

### The active rest-frame sandwich

A single Kerr–Schild chart has its spin axis along \(+z\) and its null vector in a standard form. A moving, arbitrarily oriented source is not in that chart.

For every source, at the retarded event, the evaluation chain is:

1. Binary-search the source world-line cache for the retarded event (null separation from the observer).
2. **Active Lorentz boost** into the instantaneous rest frame of the source.
3. **Active spatial rotation** that aligns the source spin with \(+z\).
4. Evaluate the untouched rest-frame Kerr tetrad / metric / Christoffel symbols at the relative position.
5. Compute geodesic acceleration \(du^\mu/d\lambda = -\Gamma^\mu{}_{\alpha\beta} u^\alpha u^\beta\) and spin transport \(dj^\mu/d\lambda = -\Gamma^\mu{}_{\alpha\beta} j^\alpha u^\beta\).
6. Inverse rotation, then inverse boost, back to the laboratory frame.
7. Sum the resulting 4-accelerations over all sources.

The boost and rotation are *active* transformations of the relative 4-vectors. They are not a passive relabelling of the tetrad. That is what keeps each source’s principal null direction aligned with the analytic Kerr–Schild chart in which \(H\) and \(L^\mu\) are known in closed form.

Because every metric evaluation is local and analytic (or a few finite differences of an analytic metric), there is no 3-D grid, no constraint solve, and no gauge evolution. Cost scales as \(O(N)\) per particle.

### What “without supercomputers” means

Traditional numerical relativity evolves ten or more metric components on an adaptive 3-D mesh, damps constraints, and maintains a gauge. Even modest binary-black-hole runs need HPC.

Here the geometry of each source is the exact Kerr solution in the frame where that solution is written. Multi-body dynamics are the retarded linear sum of the geodesic deviations those geometries produce. That is a different computational problem: 4-vector arithmetic, 4×4 Lorentz matrices, and a one-dimensional world-line cache per particle. It runs in real time in Python.

---

## 2. The quantum term: free diffusion and the Wick-rotated Schrödinger equation

There is no auxiliary wave function and no spatial grid for \(\psi\).

Each particle receives, in its instantaneous rest frame, an isotropic Gaussian displacement whose rms is

\[
\mathrm{d}r = \sqrt{\frac{\mathrm{d}\lambda}{M}}
\]

(\(\hbar = c = G = 1\)). That increment is then boosted to the laboratory frame with the same Lorentz matrix used for the classical acceleration. The boost is kinematics: it keeps the noise consistent with the particle’s 4-velocity. It is not a Nelson drift.

### Why this recovers the Wick-rotated free Schrödinger equation

The free Schrödinger equation

\[
i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi
\]

becomes, after the Wick rotation \(t \to -i\tau\), the diffusion (heat) equation

\[
\hbar\frac{\partial\psi}{\partial\tau} = \frac{\hbar^2}{2m}\nabla^2\psi.
\]

The fundamental solution is a Gaussian whose variance grows as \(\langle(\Delta x)^2\rangle = (\hbar/m)\,\mathrm{d}\tau\).

Nelson’s stochastic mechanics realises the same diffusion constant as an Itô process. With \(\hbar = 1\) and the evolution parameter taken to be proper time \(\lambda\), the rest-frame step above is exactly that process: each Cartesian component has variance \(\mathrm{d}\lambda/M\).

Three choices keep this relativistic and multi-particle without introducing a wave-function object:

1. **Proper time** is the integration parameter (the same \(\lambda\) used for the geodesic step).
2. **Isotropy in the rest frame**, then a boost, so the noise does not pick a preferred frame.
3. **Independent Wiener processes** for each particle; interactions enter only through the classical Kerr–Schild accelerations.

The ensemble of trajectories therefore satisfies the Fokker–Planck equation of the Wick-rotated free theory. The same mass \(M\) that appears as the Kerr parameter also sets the Compton-scale diffusion. That is the quantum side of the duality: a particle is a rotating black hole *and* a diffusing relativistic world-line.

Guided (pilot-wave) drift from a non-trivial \(\psi\) is deliberately omitted. The deterministic motion is the Kerr–Schild geodesic acceleration; the stochastic motion is free rest-frame diffusion.

---

## 3. Black-hole–particle duality

Every relativistic wave equation of mass \(m\) and spin \(s\) is mapped to a rotating Kerr black hole in Cartesian Kerr–Schild coordinates with parameters \(M = m\) and \(J = s\).

The three pillars (C = special relativity, G = gravitation, H = quantum mechanics) are realised together:

- **C** — 4-velocities, Lorentz matrices from the exponential of the generators, rest-frame / lab-frame sandwiches.
- **G** — exact Kerr tetrads and retarded geodesic acceleration + spin transport.
- **H** — mass-dependent free diffusion as above.

The observer is one of the particles. The renderer boosts and rotates into that observer’s rest frame, uses retarded events, and shades sources by the increment of source proper time between consecutive views.

Mergers use a horizon-entry trigger, a guarded plunge until the relative Kerr radius collapses, then a two-sided tetrad renormalization of 4-momentum and 4-spin.

---

## Layout

| File | Role |
| --- | --- |
| `Minkowski.py` | Metric, intervals, Lorentz algebra |
| `KerrSchild.py` | \(r\), \(H\), \(L^\mu\), rest-frame tetrad and metric |
| `Particle.py` | World-line cache, retarded lookup, proper-time state |
| `ParticleMergers.py` | Horizon test, plunge, tetrad-renormalized merger |
| `RelativisticRenderer.py` | Observer-centric retarded view |
| `BlackHoleOrbiter.py` / `Main.py` | Integration loop, input, Pygame |
| `UnitTests.py` | Tetrad / Lorentz tests |
| `InitializeScene.py` | Example initial data |

---

## Requirements

```text
numpy
scipy
pygame
```

Python 3.10+ recommended.

```bash
pip install numpy scipy pygame
python Main.py
```

---

## Controls (body-frame camera)

| Keys | Action |
| --- | --- |
| W A S D | Boost forward / left / back / right |
| Space / Shift | Boost up / down |
| Arrows | Pitch / yaw |
| Q / E | Roll |

The observer’s 4-spin is taken from the body-frame angular velocity so that orientation and physical spin stay consistent.

---

## References

- M. Gürses and F. Gürsey, “Lorentz covariant treatment of the Kerr–Schild geometry,” *J. Math. Phys.* **16**, 2385 (1975).
- E. Nelson, *Dynamical Theories of Brownian Motion* and the stochastic-mechanics reconstruction of the Schrödinger equation.
- Companion text: *On the Nature of Things* (Part I), included in this repository.

---

## Author

Erik Jorgensen — ScienceEngine / Black Hole Orbiter.  
The first MATLAB geodesic integrator dates to 2009. The present engine is the Kerr–Schild multi-body, retarded, spinning, merging, freely-diffusing formulation of the same idea.
