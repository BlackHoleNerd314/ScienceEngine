#!/usr/bin/env python3
"""
Black Hole Orbiter (revised).

Improvements over original:
  1. Translated from MATLAB into Python, which opens the possibility of
    making a scientifically accurate game-engine with pygame visualizations
    while keeping low level control throughout for critical features.
  2. Simulates General Relativity Numerically, by referencing a 1975 paper
    "A Lorentz Covariant Treatment of the Kerr-Schild Geometry" by Gurses and Gursey.
    It is the only known way to fully simulate the General Theory of Relativity
    by using the only coordinate system that exactly linearizes the
    Einstein Field Equations, and does so without Supercomputing Clusters.
  3. Keeps the author's complete unifying vision of a Unified Field Theory intact.
    To do this requires a paradigm shift where every particle
    is also a rotating black hole, a field quantum, and an observer.
    This means one recovers Lorentz Covariant Motion in Special Relativity,
    Newtonian Force Laws of Universal Gravitation, Quantum Mechanics, and
    Multi-Particle Theory, with both mass and spin as particle invariants.

Author: Erik Jorgensen
Date: 2026-05-30 (revised), 2009-08-22 (original)
"""

import RelativisticRenderer as Game
import UnitTests as Check

if __name__ == "__main__":
    Check.TestAll()
    Game.play()