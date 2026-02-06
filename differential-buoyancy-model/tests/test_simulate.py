import numpy as np

from differential_buoyancy_model import Params, simulate


def test_simulate_shapes():
    t, y, v = simulate(y0=-1.0, v0=0.0, t_span=(0.0, 1.0), num_points=101, p=Params())
    assert t.shape == (101,)
    assert y.shape == (101,)
    assert v.shape == (101,)
    assert np.isfinite(y).all()
    assert np.isfinite(v).all()
