import math
import numpy as np
import KerrSchild as KS
import Minkowski as mink

# -----------------------------------------------------------------------
# Change 1 & 2: Particle class with worldline cache + proper-time state
# -----------------------------------------------------------------------

class Particle:
    """
    A gravitating body.  Integration is in proper time; coordinate time
    advances at rate  dt/dλ = u^0 (the Lorentz factor).

    The worldline cache stores snapshots at every integration step,
    ordered by (monotonically increasing) coordinate time.  Retarded
    positions are found by binary search on this cache.
    """

    def __init__(self, pos4: np.ndarray, vel4: np.ndarray, spin4: np.ndarray,
                 mass: float,
                 name: str = ""):
        """
        Parameters
        ----------
        pos4   : initial 4-position (t, x, y, z).
        vel4   : initial 4-velocity (will be normalised).
        spin4  : initial 4-spin     (will be normalised)
        mass   : gravitational mass (G = c = 1).
        spin_J : 3-vector whose *magnitude* is the Kerr spin parameter a
                 and whose *direction* is the spin axis in the lab frame.
                 Examples:
                   spin_J = (0, 0, 0.5)  -> a=0.5, spin along +Z
                   spin_J = (0.3, 0, 0)  -> a=0.3, spin along +X
        name   : optional label for diagnostics.
        """
        self.pos = pos4.astype(float).copy()
        self.vel = mink.normalize_4velocity(vel4.astype(float).copy())
        self.spin = KS.Initialize_spin_vector(spin4.astype(float).copy(),self.vel)
        self.mass = mass
        self.name = name

        self.proper_time: float = 0.0
        # Cache: list of dicts, sorted by ascending coordinate time.
        self._cache: list = [self._snapshot()]
        # ancestry tracking
        self.parent_ids = []  # names of particles that formed this one
        self.children = []  # particles formed from this one

        self.birth_time = None  # coordinate time at which this particle was born
        self.death_time = None          # coordinate time at which this particle died (None = still alive)
        self.is_ghost = False           # True once it has been merged away from the live list

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    def interpolate_at(self, t_query: float):
        """
        Linearly interpolate the cached worldline at coordinate time t_query.
        Returns (pos4, vel4).  Clamps to cache endpoints if out of range.
        """
        cache = self._cache
        if t_query <= cache[0]['t']:
            return cache[0]['pos'].copy(), cache[0]['vel'].copy(), cache[0]['spin'].copy(), cache[0]['lam']
        if t_query >= cache[-1]['t']:
            return cache[-1]['pos'].copy(), cache[-1]['vel'].copy(), cache[-1]['spin'].copy(), cache[-1]['lam']

        # Binary search for bracketing entries
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

    # ------------------------------------------------------------------
    # Change 1: retarded position via cache binary search
    # ------------------------------------------------------------------

    def retarded_position(self, observer_pos4: np.ndarray,
                          tol: float = 1e-10, max_iter: int = 64):
        """
        Find the retarded 4-position and 4-velocity of this particle as
        seen from observer_pos4 by binary-searching the worldline cache.

        Solves:  c·(t_obs − t_ret) = |x_src(t_ret) − x_obs|
                 (with c = 1 in natural units)

        i.e. the separation 4-vector between source and observer is null.

        Parameters
        ----------
        observer_pos4 : 4-position of the observer at the *current* time.
        tol           : coordinate-time tolerance for the bisection.
        max_iter      : maximum bisection iterations.

        Returns
        -------
        pos_ret, vel_ret, spin_ret : 4-position, 4-velocity, 4-spin at the retarded event.
        """
        t_obs  = float(observer_pos4[0])
        x_obs  = observer_pos4[1:4]

        t_lo = self.cache_t_min
        t_hi = min(self.cache_t_max, t_obs)

        if t_hi <= t_lo:
            return self._cache[0]['pos'].copy(), self._cache[0]['vel'].copy(), self._cache[0]['spin'].copy(), self._cache[0]['lam']

        def residual(t_ret: float) -> float:
            """Positive when t_ret is too early, negative when too late."""
            pos_r, vel_r, spin_r, lam_r = self.interpolate_at(t_ret)
            dx = pos_r[1:4] - x_obs
            dist = math.sqrt(float(np.dot(dx, dx)))
            return (t_obs - t_ret) - dist

        f_lo = residual(t_lo)
        f_hi = residual(t_hi)

        # If the root is not bracketed, return the nearest endpoint
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

