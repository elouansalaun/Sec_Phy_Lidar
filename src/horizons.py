"""
src/horizons.py: dense horizons by log-polar pyramid.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange
from scipy.ndimage import maximum_filter, uniform_filter

N_FEAT = 3            # opaque horizon, vegetation horizon, opacity


# =============================================================================
# Step 1: block lattice: for each band, blocks of side p_b every p_b pixels
# =============================================================================
def block_lattice(bands_m, step_px, n_sectors: int, res: float):
    """Offsets of the block centres, level (index of p_b), band, sector, distance [m]."""
    dr, dc, lvl, band, sect, dist = [], [], [], [], [], []
    w = 360.0 / n_sectors
    for b, (lo, hi) in enumerate(zip(bands_m[:-1], bands_m[1:])):
        p = int(step_px[b])
        assert p % 2 == 1, "odd step: the block must be centred on a cell"
        k = int(np.ceil(hi / res / p)) + 1
        i, j = np.mgrid[-k:k + 1, -k:k + 1]
        r, c = (i * p).ravel(), (j * p).ravel()
        d = np.hypot(r, c) * res
        keep = (d > lo) & (d <= hi) if b > 0 else (d > 0) & (d <= hi)
        r, c, d = r[keep], c[keep], d[keep]
        az = np.degrees(np.arctan2(c, -r)) % 360.0
        dr.append(r); dc.append(c); dist.append(d)
        sect.append(np.floor((az + w / 2) / w).astype(np.int64) % n_sectors)
        band.append(np.full(r.size, b)); lvl.append(np.full(r.size, b))
    cat = lambda a, t: np.concatenate(a).astype(t)
    return (cat(dr, np.int64), cat(dc, np.int64), cat(lvl, np.int64), cat(band, np.int64),
            cat(sect, np.int64), cat(dist, np.float64))


def pyramid(s, step_px):
    """Sliding filters per level: max of z_opaque, max of z_veg, means of the opacity and ground.
    The two means are stacked: om[0] opacity, om[1] ground."""
    can = np.maximum(s.z_veg - s.z_opaque, 0.0)
    op = s.k_ext * can
    ground = (s.z_opaque < s.z_tgt).astype(np.float64)
    f = lambda fn, a: np.stack([fn(a, size=p, mode="nearest") for p in step_px]).astype(np.float32)
    return (f(maximum_filter, s.z_opaque), f(maximum_filter, s.z_veg),
            np.stack([f(uniform_filter, op), f(uniform_filter, ground)]))


# =============================================================================
# Step 2: dense kernel
# =============================================================================
@njit(parallel=True, cache=True)
def _horizons(z_opaque, zo, zv, om, rows, cols, heights, dr, dc, lvl, band, sect, dist, n_b, n_s):
    """-> horizons (n, n_h, 2, n_b, n_s) and means (n, 2, n_b, n_s): opacity, ground. Outside the
    grid, blocks are ignored: near the edge, the horizon is partial (positions never evaluated)."""
    n, nh = rows.size, heights.size
    H, W = z_opaque.shape
    hor = np.zeros((n, nh, 2, n_b, n_s), np.float32)
    opa = np.zeros((n, 2, n_b, n_s), np.float32)
    for i in prange(n):
        r0, c0 = rows[i], cols[i]
        z0 = z_opaque[r0, c0]
        ho = np.full((nh, n_b, n_s), -1e9)
        hv = np.full((nh, n_b, n_s), -1e9)
        so = np.zeros((n_b, n_s))
        ss = np.zeros((n_b, n_s))
        cnt = np.zeros((n_b, n_s))
        for m in range(dr.size):
            r, c = r0 + dr[m], c0 + dc[m]
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            L, b, s = lvl[m], band[m], sect[m]
            do, dv = zo[L, r, c] - z0, zv[L, r, c] - z0
            for ih in range(nh):
                a = (do - heights[ih]) / dist[m]
                if a > ho[ih, b, s]:
                    ho[ih, b, s] = a
                a = (dv - heights[ih]) / dist[m]
                if a > hv[ih, b, s]:
                    hv[ih, b, s] = a
            so[b, s] += om[0, L, r, c]
            ss[b, s] += om[1, L, r, c]
            cnt[b, s] += 1.0
        for b in range(n_b):
            for s in range(n_s):
                opa[i, 0, b, s] = np.log1p(so[b, s] / max(cnt[b, s], 1.0))
                opa[i, 1, b, s] = ss[b, s] / max(cnt[b, s], 1.0)
                for ih in range(nh):
                    hor[i, ih, 0, b, s] = np.arctan(ho[ih, b, s]) if cnt[b, s] > 0 else 0.0
                    hor[i, ih, 1, b, s] = np.arctan(hv[ih, b, s]) if cnt[b, s] > 0 else 0.0
    return hor, opa


class Horizons:
    """Pyramid prepared once; repeated calls on lists of positions."""

    def __init__(self, s, cfg):
        hc, sc = cfg["surrogate"]["horizons"], cfg["surrogate"]
        self.blocks = block_lattice(hc["bands_m"], hc["step_px"], sc["n_sectors"], s.grid.res)
        t0 = time.time()
        self.zo, self.zv, self.om = pyramid(s, hc["step_px"])
        self.t_prep = time.time() - t0
        self.z32 = s.z_opaque.astype(np.float32)
        self.h = np.asarray(sc["heights_m"], np.float32)
        self.nb, self.ns = len(hc["bands_m"]) - 1, sc["n_sectors"]
        self.shape = s.z_opaque.shape

    def __call__(self, rows, cols, batch=200_000):
        n = rows.size
        hor = np.empty((n, self.h.size, 2, self.nb, self.ns), np.float32)
        opa = np.empty((n, 2, self.nb, self.ns), np.float32)
        for a in range(0, n, batch):
            sl = slice(a, a + batch)
            hor[sl], opa[sl] = _horizons(self.z32, self.zo, self.zv, self.om,
                                         rows[sl].astype(np.int64), cols[sl].astype(np.int64),
                                         self.h, *self.blocks, self.nb, self.ns)
        return hor, opa

    def maps(self, lines: int = 64):
        """Full grid, in float16, computed by blocks of rows (bounded memory):
        horizons (h, 2, b, s, H, W), means (2, b, s, H, W)."""
        H, W = self.shape
        hor = np.empty((self.h.size, 2, self.nb, self.ns, H, W), np.float16)
        mean = np.empty((2, self.nb, self.ns, H, W), np.float16)
        t0 = time.time()
        for a in range(0, H, lines):
            b = min(a + lines, H)
            r, c = np.divmod(np.arange(a * W, b * W), W)
            ho, mo = self(r, c)
            hor[..., a:b, :] = np.moveaxis(ho.reshape(b - a, W, *ho.shape[1:]), (0, 1), (-2, -1))
            mean[..., a:b, :] = np.moveaxis(mo.reshape(b - a, W, *mo.shape[1:]), (0, 1), (-2, -1))
        return hor, mean, time.time() - t0


def horizons(s, cfg, rows, cols):
    return Horizons(s, cfg)(rows, cols)


# =============================================================================
# Step 3: check: cost and error against the exact horizon (B1 descriptors), per height
# =============================================================================
def main() -> None:
    import pandas as pd

    from src.site import ROOT, load_config
    from src.surrogate_baselines import BANDS_M, _descriptors, band_offsets
    from src.viewshed import prepare_surfaces

    cfg = load_config()
    hc = cfg["surrogate"]["horizons"]
    assert tuple(hc["bands_m"]) == BANDS_M, "same bands as B1 to compare"
    s = prepare_surfaces(cfg)
    hz = Horizons(s, cfg)
    print(f"{hz.blocks[0].size:,} blocks read per position (exact computation: 70,680 cells), "
          f"{hz.h.size} heights")

    # Contiguous block of the site (120 × 120), as in a dense map: the order of the positions
    # matters (cache reuse), a random draw would underestimate the gain.
    r, c = np.mgrid[400:520, 400:520]
    r, c = r.ravel().astype(np.int64), c.ravel().astype(np.int64)
    hz(r[:10], c[:10])                                     # compilation
    t0 = time.time()
    P, _ = hz(r, c)
    tp = (time.time() - t0) / r.size

    dr, dc, sect, band, dd = band_offsets(cfg)
    ground = s.z_opaque < s.z_tgt
    ns, nb = hz.ns, hz.nb
    h64 = hz.h.astype(np.float64)
    _descriptors(s.z_opaque, s.z_veg, s.k_ext, ground, r[:10], c[:10], h64, dr, dc, sect, band,
                 dd, ns, nb)
    t0 = time.time()
    E = _descriptors(s.z_opaque, s.z_veg, s.k_ext, ground, r, c, h64, dr, dc, sect, band, dd,
                     ns, nb)
    te = (time.time() - t0) / r.size
    E = E.reshape(r.size, h64.size, ns, nb, 4)             # (n, h, s, b, desc)
    ex_o = np.arctan(E[..., 0]).transpose(0, 1, 3, 2)      # (n, h, b, s)

    rows = []
    for ih in (0, h64.size - 1):
        for b in range(nb):
            e = P[:, ih, 0, b] - ex_o[:, ih, b]
            rows.append(dict(h=h64[ih], band=f"{BANDS_M[b]:g}-{BANDS_M[b + 1]:g} m",
                             step_px=hc["step_px"][b], bias_deg=np.degrees(e.mean()),
                             mae_deg=np.degrees(np.abs(e).mean()),
                             p95_deg=np.degrees(np.quantile(np.abs(e), 0.95)),
                             share_overestimated=np.mean(e > 1e-6)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"Cost ({hz.h.size} heights): pyramid {1e3 * tp:.4f} ms per position "
          f"(+ {hz.t_prep:.1f} s of filters, once for the grid); exact (B1) {1e3 * te:.3f} ms "
          f"-> ×{te / tp:.0f}")


if __name__ == "__main__":
    main()
