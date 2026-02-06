# Buoyancy Differential Model

A small Python project that simulates 1D vertical motion of an object in water using an ODE model:

- gravity
- buoyancy (based on submerged depth up to a max volume)
- quadratic drag

The default entrypoint runs a simulation and shows an interactive Plotly animation of position vs time.

## Quickstart

### 1) Create an environment + install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install -e .
```

### 2) Run

```bash
python -m buoyancy_model
```

Or, after installing editable, use the console script:

```bash
buoyancy-model
```

## Project layout

- `src/buoyancy_model/` — reusable code
- `notebooks/` — the original notebook
- `tests/` — placeholder for future tests

## Notes

The model is intentionally simple and “lumped” (e.g., drag is a single coefficient). If you want to extend it,
good next steps are: a more realistic submerged-volume geometry, linear + quadratic drag, and parameter fitting.
