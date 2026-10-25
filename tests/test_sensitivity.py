"""Tests of the sensitivity analysis (src/sensitivity.py)."""
import numpy as np
import pandas as pd
import pytest

from src.sensitivity import jaccard, keys, map_configs
from src.sensors import transmittance
from src.timely import time_remaining
from src.viewshed import ray_tau
from test_intruder import cols_mask, domain, fence

IC = dict(speed_m_s=1.5, t_fence_s=10.0, t_gate_s=3.0)


def test_foliage_equals_alpha():
    """τ is linear in k: k × f gives τ × f, and T(τ·f, α) = T(τ, α·f)."""
    rng = np.random.default_rng(0)
    H = W = 40
    z_op = np.zeros((H, W))
    z_veg = rng.uniform(0, 8, (H, W))
    k = rng.uniform(0, 0.4, (H, W))
    for f in (2.0, 4.0):
        for (r, c) in ((35, 30), (5, 38), (20, 2)):
            t1 = ray_tau(z_op, z_veg, k, 20, 20, 12.0, r, c, 1.0, 1.0)
            tf = ray_tau(z_op, z_veg, f * k, 20, 20, 12.0, r, c, 1.0, 1.0)
            assert tf == pytest.approx(f * t1, rel=1e-6)
            for a in (0.3, 1.0):
                assert transmittance(np.array([tf]), a) == pytest.approx(
                    transmittance(np.array([t1]), a * f))


def test_delay_capped_by_the_gate():
    """T_min = walking + min(t_c, t_p): reinforcing the fence alone only helps up to the gate."""
    m, ins = fence()                         # fence between columns 4 and 5, gate on row 4
    dom, *_ = domain(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, gate_rows=[4])
    walk = 9 / IC["speed_m_s"]
    for tc, tp in ((10, 3), (30, 3), (30, 30), (60, 60)):
        d = time_remaining(dom, dict(IC, t_fence_s=tc, t_gate_s=tp), "F")
        assert d[dom.source].min() == pytest.approx(walk + min(tc, tp))


def test_map_configs_and_keys():
    conf = pd.DataFrame(dict(cand=["C1", "C1", "C2"], h=[4.0, 4.0, 6.0],
                             sensor=["cam_standard", "radar_perimeter", "cam_tele"],
                             azimuth=[90.0, np.nan, 180.0], tilt=[12.3, 5.0, 8.0]))
    k = keys(conf)
    assert len(set(k)) == 3
    dst = {kk: i for i, kk in enumerate(k[[2, 0]])}          # configuration 1 has disappeared
    T, missing = map_configs([0, 1, 2], k, dst)
    assert T.tolist() == [0, 1] and missing == 1
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3) and jaccard(set(), set()) == 1.0
