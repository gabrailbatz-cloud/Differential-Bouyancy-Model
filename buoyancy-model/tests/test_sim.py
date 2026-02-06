from buoyancy_model.sim import Params, simulate

def test_simulate_runs():
    p = Params()
    t, y, v = simulate(y0=-1.0, v0=0.0, t_span=(0.0, 1.0), num_points=10, p=p)
    assert len(t) == len(y) == len(v) == 10
