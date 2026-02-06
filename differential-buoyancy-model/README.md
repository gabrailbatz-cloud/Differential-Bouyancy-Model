# Differential Buoyancy Model

A small **toy physics** simulator for 1D vertical motion in water with:
- gravity
- a clipped (depth-based) buoyancy term
- quadratic drag

It solves an ODE with `scipy.integrate.solve_ivp`, and can visualize results via **matplotlib** (static) or **plotly** (animated).

## Repo layout

- `src/differential_buoyancy_model/` — importable module + CLI
- `notebooks/` — the original notebook
- `tests/` — small sanity tests (pytest)

## Quickstart

```bash
# create a venv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# run the CLI (static plot by default)
python -m differential_buoyancy_model --y0 -1 --v0 0 --t1 10

# animated (requires plotly)
python -m differential_buoyancy_model --animate
```

## Notes / assumptions

This is intentionally simplified and not meant to be an engineering-grade buoyancy model.
In particular, the *submerged volume* is derived from `-y` and clipped to `Params.volume`.
