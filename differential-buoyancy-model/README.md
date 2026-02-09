# Differential Buoyancy Model (toy)

A small Python simulation of vertical motion near a water surface with:
- gravity
- *differential* buoyancy (depends on how submerged the object is)
- quadratic drag

It produces an interactive Plotly animation of position vs time.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python -m diff_buoyancy
```

## CLI options

```bash
python -m diff_buoyancy --y0 -1.5 --v0 0 --t1 15 --n 2000
```

- `y0 < 0` means starting below the water surface (`y = 0`).
- `--max-frames` limits animation frames to keep Plotly snappy.

## Repo layout

- `src/diff_buoyancy/` – reusable module + CLI
- `notebooks/` – original notebook(s)

## Notes / assumptions

This is intentionally simplified (e.g., “submerged volume” is approximated as `clip(-y, 0, volume)`).
Use it for intuition-building and visualization, not for designing submarines.
