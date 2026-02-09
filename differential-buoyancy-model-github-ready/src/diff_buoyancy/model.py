from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    """Physical parameters for vertical motion in water.

    Notes
    -----
    - Coordinate system: y = 0 at the water surface; y < 0 means submerged.
    - This is a toy model intended for exploration and visualization, not naval engineering.
    """
    g: float = 9.81                 # m/s^2
    rho: float = 1000.0             # kg/m^3 (water density)
    drag_coeff: float = 2.0         # lumped quadratic drag coefficient

    mass: float = 5.0               # kg
    volume: float = 1.0             # m^3 (max displaced volume)


def buoyant_force(y: float, p: Params) -> float:
    """Buoyant force (upward, +).

    Assumes the object intersects the waterline at y=0.
    If y >= 0, the object is fully out of water -> no buoyancy.
    If y < 0, the submerged 'volume' is approximated as clip(-y, 0, volume).
    """
    if y >= 0.0:
        return 0.0

    submerged_volume = float(np.clip(-y, 0.0, p.volume))
    return p.rho * submerged_volume * p.g


def rhs(t: float, state: np.ndarray, p: Params) -> np.ndarray:
    """Right-hand side of the ODE system.

    state[0] = position y
    state[1] = velocity v
    """
    y, v = state

    f_gravity = -p.mass * p.g
    f_buoyancy = buoyant_force(float(y), p)
    f_drag = -p.drag_coeff * float(v) * abs(float(v))   # quadratic drag opposes motion

    f_net = f_gravity + f_buoyancy + f_drag

    dy_dt = v
    dv_dt = f_net / p.mass
    return np.array([dy_dt, dv_dt], dtype=float)


def simulate(
    y0: float,
    v0: float,
    t_span: Tuple[float, float],
    num_points: int,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the motion and return (t, y, v)."""
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    sol = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=t_span,
        y0=np.array([y0, v0], dtype=float),
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    y = sol.y[0]
    v = sol.y[1]
    return sol.t, y, v
