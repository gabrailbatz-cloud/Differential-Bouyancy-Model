from __future__ import annotations

import argparse

from .model import Params, simulate
from .viz import animate_position_plotly


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Differential buoyancy toy model simulation.")
    p.add_argument("--y0", type=float, default=-1.0, help="Initial position (y<0 is submerged).")
    p.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    p.add_argument("--t0", type=float, default=0.0, help="Start time.")
    p.add_argument("--t1", type=float, default=10.0, help="End time.")
    p.add_argument("--n", type=int, default=1000, help="Number of time points.")
    p.add_argument("--max-frames", type=int, default=200, help="Max animation frames.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = Params()

    t, y, _v = simulate(
        y0=args.y0,
        v0=args.v0,
        t_span=(args.t0, args.t1),
        num_points=args.n,
        p=params,
    )
    animate_position_plotly(t, y, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
