"""
Tests of the dataset of the surrogate model (src/surrogate_dataset.py).

1. Consistency: the `potential` kernel gives back exactly the aggregation of `optical_depth`.
2. Flat terrain, no vegetation: the whole ground is visible, V = 1 in each bin.
3. Symmetries of the D4 group: rotating or flipping the terrain permutes the sectors.
   This is what allows the data augmentation by rotations in 5.3.
4. Monotonicity: V does not decrease when the observer goes up.
5. Split: the site is in the test set, the buffer is respected, the discs are complete.

Usage : python -m pytest tests/test_surrogate_data.py -v
"""
import numpy as np
import pytest

from src.surrogate_dataset import EXCLUDED, SPLITS, disc_bins, potential, spatial_split
from src.viewshed import optical_depth

N, RES, R = 81, 1.0, 30
RINGS, NS = [0, 10, 30], 8
H = np.array([2.0, 5.0, 12.0])
DR, DC, B, NBIN = disc_bins(R, RINGS, NS, RES)


def terrain(seed=0, veg=True):
    """Gentle relief, opaque blocks, vegetation patches: a case where everything matters."""
    rng = np.random.default_rng(seed)
    rr, cc = np.indices((N, N))
    z = 2.0 * np.sin(rr / 9.0) + 1.5 * np.cos(cc / 7.0)
    zo = z.copy()
    for _ in range(12):
        r, c, a, b = rng.integers(0, N - 6, 4) % (N - 6)
        zo[r:r + 3 + a % 4, c:c + 3 + b % 4] += rng.uniform(3, 9)
    zv = zo.copy()
    k = np.zeros_like(z)
    if veg:
        m = (rng.random((N, N)) < 0.15) & (zo == z)
        zv[m] += rng.uniform(1, 6, m.sum())
        k[m] = rng.uniform(0.05, 0.8, m.sum())
    return zo, zv, k, z + 1.0


def run(zo, zv, k, zt, obs):
    ground = zo < zt
    r = np.array([o[0] for o in obs], np.int64)
    c = np.array([o[1] for o in obs], np.int64)
    return potential(zo, zv, k, zt, ground, r, c, H, DR, DC, B, NBIN, RES).astype(np.float64)


def aggregate(zo, zv, k, zt, r0, c0, h):
    """Reference: disc of optical_depth, then sum of T per bin, in numpy."""
    tau = optical_depth(zo, zv, k, zt, r0, c0, zo[r0, c0] + h, R, RES).astype(np.float64)
    ground = zo < zt
    acc = np.zeros(NBIN.size)
    for dr, dc, b in zip(DR, DC, B):
        t = tau[r0 + dr, c0 + dc]
        if ground[r0 + dr, c0 + dc] and np.isfinite(t):
            acc[b] += np.exp(-t)
    return acc / NBIN


# -----------------------------------------------------------------------------
# 1. Consistency with the visibility engine
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_consistency_optical_depth(seed):
    zo, zv, k, zt = terrain(seed)
    obs = [(40, 40), (35, 47), (45, 33)]
    V = run(zo, zv, k, zt, obs)
    for i, (r0, c0) in enumerate(obs):
        for ih, h in enumerate(H):
            np.testing.assert_allclose(V[i, ih], aggregate(zo, zv, k, zt, r0, c0, h), atol=1e-6)


# -----------------------------------------------------------------------------
# 2. Flat terrain: everything is visible
# -----------------------------------------------------------------------------
def test_flat_everything_visible():
    z = np.zeros((N, N))
    V = run(z, z, z, z + 1.0, [(40, 40)])
    np.testing.assert_allclose(V, 1.0, atol=1e-6)
    assert NBIN.sum() == DR.size                      # each cell of the disc in one bin


def test_bin_geometry():
    # Almost uniform distribution between sectors, and both rings are filled
    n = NBIN.reshape(len(RINGS) - 1, NS)
    assert (n > 0).all()
    assert n.max(axis=1).min() / n.min(axis=1).max() < 1.1
    # North (−r) is sector 0, east (+c) sector 2
    i_n = np.flatnonzero((DR == -20) & (DC == 0))[0]
    i_e = np.flatnonzero((DR == 0) & (DC == 20))[0]
    assert B[i_n] % NS == 0 and B[i_e] % NS == 2


# -----------------------------------------------------------------------------
# 3. D4 symmetries: 90° rotation and mirror
# -----------------------------------------------------------------------------
def permute(V, perm):
    """V[..., ring·NS + s] -> sector perm[s]."""
    Vr = V.reshape(*V.shape[:-1], len(RINGS) - 1, NS)
    out = np.empty_like(Vr)
    out[..., perm] = Vr
    return out.reshape(V.shape)


def test_rotation_90():
    zo, zv, k, zt = terrain(3)
    r0, c0 = 38, 44
    V = run(zo, zv, k, zt, [(r0, c0)])
    rot = lambda a: np.rot90(a, k=-1).copy()          # clockwise: north -> east
    # (r, c) -> (c, N-1-r) for a clockwise rotation
    Vr = run(rot(zo), rot(zv), rot(k), rot(zt), [(c0, N - 1 - r0)])
    np.testing.assert_allclose(Vr, permute(V, (np.arange(NS) + 2) % NS), atol=1e-5)


def test_mirror():
    zo, zv, k, zt = terrain(4)
    r0, c0 = 42, 37
    V = run(zo, zv, k, zt, [(r0, c0)])
    fl = lambda a: a[:, ::-1].copy()                  # east <-> west: azimuth -> −azimuth
    Vm = run(fl(zo), fl(zv), fl(k), fl(zt), [(r0, N - 1 - c0)])
    np.testing.assert_allclose(Vm, permute(V, (NS - np.arange(NS)) % NS), atol=1e-5)


# -----------------------------------------------------------------------------
# 4. Monotonicity in height
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [5, 6])
def test_monotone_height_opaque(seed):
    """Without vegetation, the ray goes up at every point when z_obs goes up: V is non-decreasing."""
    zo, zv, k, zt = terrain(seed, veg=False)
    V = run(zo, zv, k, zt, [(40, 40), (33, 50)])
    assert np.all(np.diff(V, axis=1) >= -1e-6)


def test_vegetation_breaks_monotonicity():
    """With vegetation, monotonicity is NOT guaranteed: a ray fully under the canopy travels
    √(1 + slope²) m per horizontal metre inside it, and the slope grows with h (§8.2).
    The network must therefore not be constrained to be monotone in h."""
    z = np.zeros((N, N))
    zv, k = z.copy(), z.copy()
    zv[40, 45:48] = 30.0                              # tall canopy column, crossed through and through
    k[40, 45:48] = 0.5
    zt = z + 1.0
    V = run(z, zv, k, zt, [(40, 40)])
    b = B[np.flatnonzero((DR == 0) & (DC == 10))[0]]
    assert V[0, 1, b] < V[0, 0, b]                     # h = 5 m sees less than h = 2 m


# -----------------------------------------------------------------------------
# 5. Spatial split
# -----------------------------------------------------------------------------
class _G:
    xmin, ymax, res = 0.0, 400.0, 1.0


class _S:
    z_tgt = np.zeros((400, 400))
    grid = _G()


def test_split():
    cfg = {"project": {"random_seed": 0},
           "surrogate": {"radius_m": 60, "test": {"margin_m": 20, "buffer_m": 15},
                         "validation": {"tile_m": 50, "share": 0.2}}}
    split = spatial_split(_S(), cfg, (180.0, 180.0, 220.0, 220.0))   # 40 m site in the centre
    r, c = np.nonzero(split != EXCLUDED)
    assert r.min() >= 60 and r.max() < 340 and c.min() >= 60 and c.max() < 340   # full disc
    # the site is entirely in the test set
    assert (split[180:220, 180:220] == SPLITS["test"]).all()
    # no training or validation cell within 15 m of the test block
    tr, tc = np.nonzero(split == SPLITS["test"])
    a, b = np.nonzero(split <= SPLITS["validation"])
    dr = np.maximum(np.maximum(tr.min() - a, a - tr.max()), 0)
    dc = np.maximum(np.maximum(tc.min() - b, b - tc.max()), 0)
    assert np.maximum(dr, dc).min() > 15
    assert (split == SPLITS["validation"]).any()
