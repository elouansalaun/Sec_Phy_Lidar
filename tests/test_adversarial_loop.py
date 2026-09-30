"""Tests of the adversarial loop (src/adversarial_loop.py, formulation §9.2): brute force on toy instances."""
import itertools

import numpy as np
import pytest
from scipy import sparse

from src.adversarial_loop import adversarial_loop, coefficients, level
from src.instance import Instance, cost
from src.intruder import corridor_threshold, detection
from src.timely import time_remaining
from test_intruder import cols_mask, domain, fence

IC = dict(speed_m_s=1.5, t_fence_s=10.0, t_gate_s=3.0, tau_obs_s=1.0)


def toy(seed, nJ=7):
    rng = np.random.default_rng(seed)
    m, ins = fence(H=7, W=8)
    ins[:, :] = False; ins[:, 4:] = True
    dom, *_ = domain(m, cols_mask(7, 8, 0), cols_mask(7, 8, 7), inside=ins, gate_rows=[3])
    D = rng.random((nJ, dom.n)) * (rng.random((nJ, dom.n)) < 0.45)
    D[D > 0.85] = 1.0
    A = sparse.csr_matrix(D)
    n_int = int(dom.inside.sum())
    order = np.r_[np.flatnonzero(dom.inside), np.flatnonzero(~dom.inside)]
    assert (order[:n_int] == np.flatnonzero(dom.inside)).all()
    sup = rng.integers(0, 4, nJ)                             # 4 supports, 2 heights per location
    inst = Instance(A=sparse.csr_matrix(D[:, dom.inside]), w=np.ones(n_int),
                    support=sup, kind=rng.integers(0, 2, nJ),
                    cost_sensor=np.round(rng.uniform(0.5, 2, nJ), 2),   # to the cent, like the site
                    site=np.array([0, 0, 1, 1]), cost_support=np.round(rng.uniform(0.2, 1, 4), 2),
                    capacity=np.array([2, 1]))
    return dom, A, inst


def brute_force(dom, A, inst, d, theta, T_R, fusion):
    from src.instance import violations
    best = np.inf
    for k in range(inst.n_configs + 1):
        for T in itertools.combinations(range(inst.n_configs), k):
            T = np.array(T, np.int64)
            if violations(inst, T, 1e6):
                continue
            P = (detection(A, T)[0] if fusion == "product"
                 else (A[T].toarray().max(axis=0) if T.size else np.zeros(dom.n)))
            p = corridor_threshold(dom, np.where(d >= T_R, P, 0.0), "F")[0]
            if p >= theta - 1e-9:
                best = min(best, cost(inst, T))
    return best


@pytest.mark.parametrize("fusion", ["max", "product"])
def test_level_equivalence(fusion):
    rng = np.random.default_rng(3)
    D = rng.random((6, 40)) * (rng.random((6, 40)) < 0.6)
    D[0, :5] = 1.0
    A = sparse.csr_matrix(D)
    for theta in (0.3, 0.6, 0.9):
        C = coefficients(A, theta, fusion)
        for _ in range(20):
            T = np.flatnonzero(rng.random(6) < 0.5)
            P = detection(A, T)[0] if fusion == "product" else (D[T].max(axis=0) if T.size else 0 * D[0])
            assert ((level(C, T) >= 1 - 1e-9) == (P >= theta - 1e-9)).all()


@pytest.mark.parametrize("fusion", ["max", "product"])
def test_equals_brute_force(fusion):
    n_ok = 0
    for seed in range(12):
        dom, A, inst = toy(seed)
        d = time_remaining(dom, IC, "F")
        for theta, T_R in ((0.6, 0.0), (0.8, 0.0), (0.6, 6.0)):
            ref = brute_force(dom, A, inst, d, theta, T_R, fusion)
            r = adversarial_loop(inst, dom, A, d, theta, T_R, "F", "cost", n_paths=3,
                                 fusion=fusion, max_time_s=60)
            if np.isinf(ref):
                assert r.status == "infeasible"
            else:
                assert r.status == "optimal" and r.value == pytest.approx(ref)
                n_ok += 1
    assert n_ok >= 10                                      # the feasible cases are covered


def test_cost_step():
    from src.adversarial_loop import cost_step
    _, _, inst = toy(0)
    assert cost_step(inst) == 0.01


def test_reusable_pool():
    """A pool filled by one case does not change the optimum of another case."""
    dom, A, inst = toy(5)
    d = time_remaining(dom, IC, "F")
    pool = []
    adversarial_loop(inst, dom, A, d, 0.6, 0.0, "F", "cost", fusion="max", pool=pool)
    assert pool
    for theta, T_R in ((0.8, 0.0), (0.6, 6.0)):
        a = adversarial_loop(inst, dom, A, d, theta, T_R, "F", "cost", fusion="max")
        b = adversarial_loop(inst, dom, A, d, theta, T_R, "F", "cost", fusion="max",
                             pool=list(pool))
        assert a.status == b.status
        if a.status == "optimal":
            assert a.value == pytest.approx(b.value)
