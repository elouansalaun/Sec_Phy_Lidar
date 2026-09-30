"""Tests of the timely detection (src/timely.py, formulation §9.1) on synthetic domains."""
import numpy as np
import pytest
from scipy import sparse

from src.intruder import corridor_threshold, detection
from src.timely import (ceiling, evaluate_timely, fastest_path, mask_timely, path_value,
                        remaining_along, time_remaining)
from test_intruder import cols_mask, domain, fence

IC = dict(speed_m_s=1.5, t_fence_s=10.0, t_gate_s=3.0, tau_obs_s=1.0)
V = IC["speed_m_s"]


def line(W=12, fence_col=None, gate=False):
    m = np.ones((1, W), bool)
    ins = None
    if fence_col is not None:
        ins = np.zeros((1, W), bool); ins[:, fence_col:] = True
    return domain(m, cols_mask(1, W, 0), cols_mask(1, W, W - 1), inside=ins,
                  gate_rows=[0] if gate else None)


# ---------------------------------------------------------------------------- d(v)
def test_time_remaining_line():
    W = 12
    dom, r, c = line(W)
    d = time_remaining(dom, IC, "F")
    assert d == pytest.approx((W - 1 - c) / V)
    assert d[dom.target] == pytest.approx(0.0)


def test_time_remaining_fence_and_gate():
    W = 12
    dom, _, c = line(W, fence_col=5)
    assert time_remaining(dom, IC, "F")[c == 0][0] == pytest.approx((W - 1) / V + IC["t_fence_s"])
    dom, _, c = line(W, fence_col=5, gate=True)
    for pr in ("F", "G"):
        assert time_remaining(dom, IC, pr)[c == 0][0] == pytest.approx((W - 1) / V + IC["t_gate_s"])


def test_G_unreachable_without_gate():
    """Fence without a gate: the opportunist cannot reach the target, d = inf outside."""
    dom, r, c = domain(*fence()[:1], cols_mask(9, 10, 0), cols_mask(9, 10, 9),
                       inside=fence()[1])
    d = time_remaining(dom, IC, "G")
    assert np.isinf(d[~dom.inside]).all() and np.isfinite(d[dom.inside]).all()


def test_remaining_along_equals_d_on_the_fastest():
    """Along the fastest path, the real remaining time is exactly d."""
    m, ins = fence()
    dom, *_ = domain(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, gate_rows=[4])
    for pr in ("F", "G"):
        d = time_remaining(dom, IC, pr)
        ch = fastest_path(dom, IC, pr)
        assert remaining_along(dom, IC, pr, ch) == pytest.approx(d[ch])


def test_d_F_below_d_G():
    m, ins = fence()
    dom, *_ = domain(m, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins, gate_rows=[4])
    assert (time_remaining(dom, IC, "F") <= time_remaining(dom, IC, "G") + 1e-12).all()


# ---------------------------------------------------------------------------- p̃*
def test_late_detection_useless():
    """Line: a detection 1 s before the target no longer counts as soon as T_R > 1 s."""
    W = 12
    dom, _, c = line(W)
    d = time_remaining(dom, IC, "F")
    P = np.where(c == W - 2, 0.9, 0.0)                  # sensor just before the target
    sigma = -np.log1p(-P)
    assert evaluate_timely(dom, P, sigma, IC, 0.5, {"F": d, "G": d})["p_tilde_F"] == pytest.approx(0.9)
    assert evaluate_timely(dom, P, sigma, IC, 1.0, {"F": d, "G": d})["p_tilde_F"] == 0.0
    P = np.where(c == 1, 0.9, 0.0)                      # early detection: useful up to d(1)
    r = evaluate_timely(dom, P, -np.log1p(-P), IC, d[c == 1][0], {"F": d, "G": d})
    assert r["p_tilde_F"] == pytest.approx(0.9)


def instances(n=20, seed=0):
    rng = np.random.default_rng(seed)
    m, ins = fence()
    for _ in range(n):
        mm = m & (rng.random(m.shape) > 0.1)
        mm[:, 0] = mm[:, -1] = mm[4, :] = True
        dom, *_ = domain(mm, cols_mask(9, 10, 0), cols_mask(9, 10, 9), inside=ins,
                         gate_rows=[4])
        P = rng.random(dom.n) * (rng.random(dom.n) < 0.6)
        d = {pr: time_remaining(dom, IC, pr) for pr in ("F", "G")}
        yield dom, P, -np.log1p(-P), d


def test_zero_T_R_gives_back_p_star():
    for dom, P, sigma, d in instances():
        r = evaluate_timely(dom, P, sigma, IC, 0.0, d)
        for pr in ("F", "G"):
            assert r[f"p_tilde_{pr}"] == pytest.approx(corridor_threshold(dom, P, pr)[0])


def test_proposition_T_R_above_T_min():
    """T_R > T_min ⇒ p̃* = 0 whatever the layout (formulation §9.1)."""
    for dom, P, sigma, d in instances():
        for pr in ("F", "G"):
            T_min = d[pr][dom.source].min()
            r = evaluate_timely(dom, np.maximum(P, 0.99), sigma + 5, IC, T_min + 1e-6, d)
            assert r[f"p_tilde_{pr}"] == 0.0


def test_monotonicity_and_bracket():
    for dom, P, sigma, d in instances():
        for pr in ("F", "G"):
            prev = np.inf
            for T_R in np.linspace(0, d[pr][dom.source].min() + 2, 15):
                r = evaluate_timely(dom, P, sigma, IC, T_R, d)
                assert r[f"p_tilde_{pr}"] <= prev + 1e-12
                assert r[f"p_tilde_{pr}"] <= r[f"p_tilde_ub_{pr}"] + 1e-12
                prev = r[f"p_tilde_{pr}"]


def test_mask_and_path_value():
    P, s, d = np.array([0.5, 0.8, 0.9]), np.array([1.0, 2.0, 3.0]), np.array([5.0, 3.0, 1.0])
    Pm, sm = mask_timely(P, s, d, 3.0)
    assert Pm.tolist() == [0.5, 0.8, 0.0] and sm.tolist() == [1.0, 2.0, 0.0]
    ch, left = np.array([0, 1, 2]), np.array([5.0, 3.0, 1.0])
    assert path_value(P, ch, left, 3.0) == 0.8
    assert path_value(P, ch, left, 6.0) == 0.0


def test_ceiling_equals_full_detection():
    rng = np.random.default_rng(1)
    D = rng.random((6, 15)) * (rng.random((6, 15)) < 0.5)
    D[2, 3] = 1.0                                        # certain detection: σ = inf, P = 1
    A = sparse.csr_matrix(D)
    P, s = ceiling(A)
    P_ref, s_ref = detection(A, np.arange(6))
    assert P == pytest.approx(P_ref) and s == pytest.approx(s_ref)
