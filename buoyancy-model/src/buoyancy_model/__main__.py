from __future__ import annotations

from buoyancy_model.sim import Params, animate_position_plotly, simulate


def main() -> None:
    p = Params()

    # Initial conditions (y < 0 means below water surface)
    y0 = -1.0
    v0 = 0.0

    # Time settings
    t_span = (0.0, 10.0)
    num_points = 1000

    t, y, _v = simulate(y0=y0, v0=v0, t_span=t_span, num_points=num_points, p=p)
    animate_position_plotly(t, y)


if __name__ == "__main__":
    main()
