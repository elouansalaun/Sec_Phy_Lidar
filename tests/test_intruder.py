"""Tests of the intruder metrics (src/intruder.py, formulation §7.8) on synthetic domains."""
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from src.intruder import Domain, adjacency, corridor_threshold, exposure
from src.intruder_domain import build_edges
from src.site import Grid

IC = dict(speed_m_s=1.5, t_fence_s=10.0, t_gate_s=3.0)


def domain(mask, source, target, inside=None, res=1.0, gate_rows=None):
    """Domain on a grid: `source`/`target`/`inside` are (H, W) masks."""
    H, W = mask.shape
    g = Grid(crs="EPSG:2950", xmin=0.0, ymin=0.0, xmax=W * res, ymax=H * res, res=res)
    rows, cols = np.nonzero(mask)
    ins = np.zeros(rows.size, bool) if inside is None else inside[rows, cols]
    u, v, l, fr = build_edges(g, rows, cols, ins)
    gate = np.full(u.size, -1, np.int16)
    if gate_rows is not None:             # crossings whose exterior cell lies on these rows
        ext = np.where(~ins[u], u, v)
        gate[fr & np.isin(rows[ext], gate_rows)] = 0
    dom = Domain(rows=rows, cols=cols, inside=ins, source=source[rows, cols],
                 target=target[rows, cols], u=u, v=v, length=l, crosses=fr, gate=gate)
    return dom, rows, cols


def cols_mask(H, W, c):
    m = np.zeros((H, W), bool)
    m[:, c] = True
    return m


def p_star_bisection(dom, P, profile):
    """Brute-force reference: smallest t such that {P ≤ t} connects source and target."""
    mask = dom.edges(profile)
    for t in np.unique(P):
        ok = P <= t
        keep = mask & ok[dom.u] & ok[dom.v]
        _, lab = connected_components(adjacency(dom, keep), directed=False)
        if np.isin(lab[dom.source & ok], lab[dom.target & ok]).any():
            return t
    return np.inf


# ---------------------------------------------------------------------------- p*
def test_corridor_between_two_walls():
    """P = 0.9 everywhere except a corridor at 0.3: p* = 0.3 and the corridor stays in the line."""
    H, W = 7, 12
    dom, r, c = domain(np.ones((H, W), bool), cols_mask(H, W, 0), cols_mask(H, W, W - 1))
    P = np.where(r == 3, 0.3, 0.9)
    p, path = corridor_threshold(dom, P, "F")
    assert p == pytest.approx(0.3)
    assert (r[path] == 3).all() and P[path].max() == pytest.approx(0.3)


def test_union_find_equals_bisection():
    rng = np.random.default_rng(0)
    for _ in range(20):
        H, W = 9, 11
        mask = rng.random((H, W)) > 0.15
        mask[:, 0] = mask[:, -1] = True
        dom, *_ = domain(mask, cols_mask(H, W, 0), cols_mask(H, W, W - 1))
        P = rng.random(dom.n).round(2)
        ref = p_star_bisection(dom, P, "F")
        if not np.isfinite(ref):
            continue
        assert corridor_threshold(dom, P, "F")[0] == pytest.approx(ref)


def test_source_equals_target():
    """A node that is both start and target: p* = its own detection, Λ = 0."""
    m = np.ones((3, 3), bool)
    s = np.zeros((3, 3), bool); s[1, 1] = True
    dom, *_ = domain(m, s, s)
    P = np.full(dom.n, 0.4)
    assert corridor_threshold(dom, P, "F")[0] == pytest.approx(0.4)
    assert exposure(dom, -np.log1p(-P), "F", IC, 1.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_bottleneck():
    """Two rooms connected by a single cell: the corridor goes through the bottleneck."""
    m = np.zeros((5, 9), bool)
    m[:, :4] = m[:, 5:] = True
    m[2, 4] = True
    dom, r, c = domain(m, cols_mask(5, 9, 0), cols_mask(5, 9, 8))
    P = np.zeros(dom.n)
    neck = np.flatnonzero((r == 2) & (c == 4))[0]
    P[neck] = 0.7
    p, path = corridor_threshold(dom, P, "F")
    assert p == pytest.approx(0.7) and neck in path
    assert neck in exposure(dom, -np.log1p(-P), "F", IC, 1.0)[1]


def test_no_path():
    m = np.ones((3, 5), bool)
    m[:, 2] = False
    dom, *_ = domain(m, cols_mask(3, 5, 0), cols_mask(3, 5, 4))
    with pytest.raises(ValueError):
        corridor_threshold(dom, np.zeros(dom.n), "F")


# ---------------------------------------------------------------------------- Λ
def test_exposure_straight_line():
    """Uniform σ on a line of W cells: Λ = σ·(W − 1)·res / (v·τ)."""
    W, sig, tau = 20, 0.05, 2.0
    m = np.ones((1, W), bool)
    dom, *_ = domain(m, cols_mask(1, W, 0), cols_mask(1, W, W - 1))
    lam, _ = exposure(dom, np.full(dom.n, sig), "F", IC, tau)
    assert lam == pytest.approx(sig * (W - 1) / (IC["speed_m_s"] * tau))


def test_resolution_invariance():
    """Linear σ(x): Λ identical at 1 m and 0.5 m (exact trapezoids); the per-cell product is not."""
    L = 20.0

    def run(res):
        n = int(L / res) + 1
        m = np.ones((1, n), bool)
        dom, r, c = domain(m, cols_mask(1, n, 0), cols_mask(1, n, n - 1), res=res)
        x = c * res
        sigma = 0.01 + 0.004 * x
        lam, path = exposure(dom, sigma, "F", IC, 1.0)
        prod = 1 - np.prod(np.exp(-sigma[path]))          # one observation per cell
        return lam, prod

    (l1, p1), (l2, p2) = run(1.0), run(0.5)
    assert l1 == pytest.approx(l2, rel=1e-9)
    assert abs(p2 - p1) > 0.1                              # the product depends on the grid


def test_crossing_time():
    """Two cells on either side of the fence: Λ = σ·(ℓ/v + t_c)/τ, or t_p at the gate."""
    m = np.ones((1, 2), bool)
    ins = np.array([[False, True]])
    s = np.array([[True, False]]); t = ~s
    sig = 0.2
    dom, *_ = domain(m, s, t, inside=ins)
    lam, _ = exposure(dom, np.full(2, sig), "F", IC, 1.0)
    assert lam == pytest.approx(sig * (1 / IC["speed_m_s"] + IC["t_fence_s"]))
    dom_g, *_ = domain(m, s, t, inside=ins, gate_rows=[0])
    lam_g, _ = exposure(dom_g, np.full(2, sig), "G", IC, 1.0)
    assert lam_g == pytest.approx(sig * (1 / IC["speed_m_s"] + IC["t_gate_s"]))


# ---------------------------------------------------------------------------- profiles
def fence(H=9, W=10, walls=False):
    """Vertical fence between columns 4 and 5, gate on row 4."""
    m = np.ones((H, W), bool)
    ins = np.zeros((H, W), bool); ins[:, 5:] = True
    if walls:                                  # the fence can only be crossed at the gate
        m[:, 4] = False; m[4, 4] = True
    return m, ins


def test_gate_only_passage():
    m, ins = fence(walls=True)
    H, W = m.shape
    rng = np.random.default_rng(1)
    dom, r, c = domain(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins,
                       gate_rows=[4])
    # With walls, every crossing leaves from the gate cell: F and G coincide
    assert ((dom.gate >= 0) == dom.crosses).all()
    for _ in range(10):
        P = rng.random(dom.n)
        sig = -np.log1p(-P)
        assert corridor_threshold(dom, P, "F")[0] == corridor_threshold(dom, P, "G")[0]
        assert exposure(dom, sig, "F", IC, 1.0)[0] == pytest.approx(
            exposure(dom, sig, "G", IC, 1.0)[0])


def test_F_below_G():
    rng = np.random.default_rng(2)
    m, ins = fence()
    H, W = m.shape
    dom, *_ = domain(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins, gate_rows=[4])
    for _ in range(20):
        P = rng.random(dom.n) * 0.9
        sig = -np.log1p(-P)
        assert corridor_threshold(dom, P, "F")[0] <= corridor_threshold(dom, P, "G")[0]
        assert exposure(dom, sig, "F", IC, 1.0)[0] <= exposure(dom, sig, "G", IC, 1.0)[0] + 1e-9


def test_profile_G_goes_through_the_gate():
    m, ins = fence()
    H, W = m.shape
    dom, r, c = domain(m, cols_mask(H, W, 0), cols_mask(H, W, W - 1), inside=ins, gate_rows=[4])
    P = np.full(dom.n, 0.2)
    _, path = exposure(dom, -np.log1p(-P), "G", IC, 1.0)
    side = dom.inside[path]
    k = np.flatnonzero(side[1:] != side[:-1])
    assert k.size == 1 and r[path[k[0]]] == 4 == r[path[k[0] + 1]]


def test_certain_detection_blocks():
    """σ = ∞ on a whole column: Λ = ∞ (every path is detected for sure)."""
    H, W = 4, 6
    dom, r, c = domain(np.ones((H, W), bool), cols_mask(H, W, 0), cols_mask(H, W, W - 1))
    sig = np.where(c == 3, np.inf, 0.1)
    lam, path = exposure(dom, sig, "F", IC, 1.0)
    assert lam == np.inf and path.size == 0
