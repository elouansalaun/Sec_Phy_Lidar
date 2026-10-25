"""
Tests of the dense horizons (src/horizons.py) and of the augmentation of the physics network.

1. Blocks of 1 px: the pyramid gives back EXACTLY the horizons of B1, at every height.
2. Tiling: exact within each band; ≥ 90 % of the disc covered once (band boundaries).
3. D4 equivariance: rotating the terrain rotates the horizon maps and permutes the sectors,
   exactly like the labels (`transform` = `surrogate.d4`).

"""
import numpy as np
import pytest

from src.horizons import Horizons, block_lattice
from src.surrogate import d4
from src.surrogate_baselines import _descriptors
from src.surrogate_physics import transform

N, NS = 101, 8
BANDS = [0, 5, 12, 25, 40]
H = [2.0, 7.0]


class _G:
    res = 1.0


class _S:
    def __init__(self, zo, zv, k, zt):
        self.z_opaque, self.z_veg, self.k_ext, self.z_tgt, self.grid = zo, zv, k, zt, _G()


def cfg(step):
    return {"surrogate": {"horizons": {"bands_m": BANDS, "step_px": step},
                          "n_sectors": NS, "heights_m": H}}


def terrain(seed=0):
    rng = np.random.default_rng(seed)
    zo = rng.uniform(0, 1.5, (N, N))
    zo[rng.random((N, N)) < 0.04] += rng.uniform(3, 10)
    zv = zo + np.where(rng.random((N, N)) < 0.1, rng.uniform(1, 5, (N, N)), 0.0)
    k = np.where(zv > zo, 0.3, 0.0)
    return _S(zo, zv, k, zo + 1.0)


def exact_b1(s, r, c):
    """Exact horizons (tangents) by the same logic as B1, on the same bands."""
    R = BANDS[-1]
    dr, dc = np.mgrid[-R:R + 1, -R:R + 1]
    dr, dc = dr.ravel(), dc.ravel()
    d = np.hypot(dr, dc)
    keep = (dr * dr + dc * dc <= R * R) & (d > 0)
    dr, dc, d = dr[keep], dc[keep], d[keep]
    az = np.degrees(np.arctan2(dc, -dr)) % 360
    sect = (np.floor((az + 22.5) / 45).astype(np.int64)) % NS
    band = np.searchsorted(np.asarray(BANDS[1:-1], float), d, side="left")
    E = _descriptors(s.z_opaque, s.z_veg, s.k_ext, s.z_opaque < s.z_tgt, r, c,
                     np.asarray(H), dr.astype(np.int64), dc.astype(np.int64), sect, band, d,
                     NS, len(BANDS) - 1)
    return E.reshape(r.size, len(H), NS, len(BANDS) - 1, 4)   # (n, h, s, b, desc)


def test_unit_blocks_equal_b1():
    s = terrain(0)
    r = np.array([50, 45, 60], np.int64)
    c = np.array([50, 58, 47], np.int64)
    hor, mean = Horizons(s, cfg([1, 1, 1, 1]))(r, c)            # (n, h, 2, b, s)
    E = exact_b1(s, r, c)
    np.testing.assert_allclose(hor[:, :, 0], np.arctan(E[..., 0]).transpose(0, 1, 3, 2), atol=1e-5)
    np.testing.assert_allclose(hor[:, :, 1], np.arctan(E[..., 1]).transpose(0, 1, 3, 2), atol=1e-5)
    np.testing.assert_allclose(mean[:, 1], E[:, 0, :, :, 3].transpose(0, 2, 1), atol=1e-5)  # ground


def coverage(bands, step, M):
    dr, dc, lvl, *_ = block_lattice(bands, step, NS, 1.0)
    tot = np.zeros((2 * M + 1,) * 2, int)
    per_band = [np.zeros_like(tot) for _ in step]
    for r, c, L in zip(dr, dc, lvl):
        p = step[L] // 2
        sl = (slice(M + r - p, M + r + p + 1), slice(M + c - p, M + c + p + 1))
        tot[sl] += 1
        per_band[L][sl] += 1
    return tot, per_band


@pytest.mark.parametrize("step", [[1, 3, 3, 5], [1, 3, 5, 9]])
def test_tiling_within_each_band(step):
    """Within a band, the blocks never overlap (lattice of step p, blocks of side p)."""
    _, per_band = coverage(BANDS, step, 60)
    assert all(pb.max() == 1 for pb in per_band)


def test_global_coverage():
    """Default configuration: ≥ 90 % of the disc covered exactly once (the rest, at the band
    boundaries, is overlapped or not covered: documented approximation)."""
    tot, _ = coverage([0, 10, 25, 50, 100, 150], [1, 3, 5, 9, 15], 170)
    rr, cc = np.mgrid[-170:171, -170:171]
    d = np.hypot(rr, cc)
    disc = (d > 0) & (d <= 150)
    assert (tot[disc] == 1).mean() > 0.9
    assert (tot[(d > 0) & (d <= 9)] == 1).all()                  # band 0 (step of 1): exact


def test_d4_equivariance():
    s = terrain(1)
    r = np.array([50], np.int64)
    c = np.array([44], np.int64)
    base, _ = Horizons(s, cfg([1, 3, 5, 5]))(r, c)
    rot = lambda a: np.rot90(a, k=-1).copy()
    sr = _S(rot(s.z_opaque), rot(s.z_veg), rot(s.k_ext), rot(s.z_tgt))
    hr, _ = Horizons(sr, cfg([1, 3, 5, 5]))(np.array([c[0]]), np.array([N - 1 - r[0]]))
    perm = (np.arange(NS) + 2) % NS
    expected = np.empty_like(base)
    expected[..., perm] = base
    np.testing.assert_allclose(hr, expected, atol=1e-5)


def test_transform_equals_d4():
    rng = np.random.default_rng(3)
    x = rng.random((5, 16, 16)).astype(np.float32)
    y = rng.random((5 * 2 * 8, 16, 16)).astype(np.float32)
    for k in range(4):
        for m in (False, True):
            _, yd = d4(x, y, k, m, 8, (5, 2, 8))
            yt = transform(y.reshape(5, 2, 8, 16, 16), k, m, 8, sector=True)
            np.testing.assert_array_equal(yd, yt.reshape(y.shape))
