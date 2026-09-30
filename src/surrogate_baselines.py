"""
src/surrogate_baselines.py: baselines B0, B1 and B2 of the surrogate model (§8.5, step 5.3).

B0: mean of the training labels per stratum (roof, open ground, ground under vegetation),
    for each (height, ring, sector).
B1: gradient boosting on hand-crafted local descriptors. For each 45° sector and each
    distance band (0-10, 10-25, 25-50, 50-100, 100-150 m):
      - tangent of the opaque horizon angle      max (z_opaque − z_obs) / d
      - tangent of the vegetation horizon angle  max (z_veg − z_obs) / d
      - mean vertical optical depth              mean of k · (z_veg − z_opaque)
      - ground share
    One row = (observer, height, sector, target ring); the sector is not a variable: the
    model is rotation invariant by construction.
B2: the same boosting, on the horizons of the log-polar pyramid (src/horizons.py), computed at
    each height: the gap B1 − B2 isolates the cost of the block approximation.

Outputs (surrogate.out_dir): baselines_test.npz (predictions), baselines.csv (metrics),
                             b2.joblib (B2 model, reused by src.surrogate_catalog)
Usage : python -m src.surrogate_baselines
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.ensemble import HistGradientBoostingRegressor

from src.site import ROOT, load_config
from src.surrogate_dataset import disc_bins
from src.surrogate_metrics import bin_areas, report, site_interior
from src.viewshed import prepare_surfaces

BANDS_M = (0.0, 10.0, 25.0, 50.0, 100.0, 150.0)
N_DESC = 4


# =============================================================================
# Step 1: descriptors per (observer, height, sector)
# =============================================================================
def band_offsets(cfg):
    sc, res = cfg["surrogate"], cfg["site"]["resolution_m"]
    dr, dc, b, _ = disc_bins(sc["radius_m"], sc["rings_m"], sc["n_sectors"], res)
    sect = b % sc["n_sectors"]
    d = np.hypot(dr, dc) * res
    band = np.searchsorted(np.asarray(BANDS_M[1:-1]), d, side="left")
    return dr, dc, sect, band, d


@njit(parallel=True, cache=True)
def _descriptors(z_opaque, z_veg, k_ext, ground, obs_r, obs_c, heights, dr, dc, sect, band, d,
                 n_sect, n_band):
    n, nh = obs_r.size, heights.size
    out = np.zeros((n, nh, n_sect, n_band * 4), np.float32)
    for i in prange(n):
        r0, c0 = obs_r[i], obs_c[i]
        t_o = np.full((nh, n_sect, n_band), -1e9)
        t_v = np.full((nh, n_sect, n_band), -1e9)
        s_k = np.zeros((n_sect, n_band))
        s_s = np.zeros((n_sect, n_band))
        cnt = np.zeros((n_sect, n_band))
        for m in range(dr.size):
            r, c = r0 + dr[m], c0 + dc[m]
            s, b = sect[m], band[m]
            zo, zv = z_opaque[r, c], z_veg[r, c]
            for ih in range(nh):
                z = z_opaque[r0, c0] + heights[ih]
                a = (zo - z) / d[m]
                if a > t_o[ih, s, b]:
                    t_o[ih, s, b] = a
                a = (zv - z) / d[m]
                if a > t_v[ih, s, b]:
                    t_v[ih, s, b] = a
            s_k[s, b] += k_ext[r, c] * (zv - zo)
            s_s[s, b] += 1.0 if ground[r, c] else 0.0
            cnt[s, b] += 1.0
        for ih in range(nh):
            for s in range(n_sect):
                for b in range(n_band):
                    out[i, ih, s, 4 * b] = t_o[ih, s, b]
                    out[i, ih, s, 4 * b + 1] = t_v[ih, s, b]
                    out[i, ih, s, 4 * b + 2] = s_k[s, b] / cnt[s, b]
                    out[i, ih, s, 4 * b + 3] = s_s[s, b] / cnt[s, b]
    return out


def descriptors(s, cfg, rows, cols, batch=5000):
    dr, dc, sect, band, d = band_offsets(cfg)
    ground = s.z_opaque < s.z_tgt
    h = np.asarray(cfg["surrogate"]["heights_m"], np.float64)
    ns, nb = cfg["surrogate"]["n_sectors"], len(BANDS_M) - 1
    out = np.empty((rows.size, h.size, ns, nb * N_DESC), np.float32)
    for a in range(0, rows.size, batch):
        sl = slice(a, a + batch)
        out[sl] = _descriptors(s.z_opaque, s.z_veg, s.k_ext, ground, rows[sl].astype(np.int64),
                               cols[sl].astype(np.int64), h, dr, dc, sect, band, d, ns, nb)
    return out


def table(D, stratum, heights, n_rings):
    """(n, h, s, f) -> rows (n·h·s·ring, f + 3): descriptors, stratum, h, target ring."""
    n, nh, ns, nf = D.shape
    X = np.repeat(D.reshape(n * nh * ns, nf), n_rings, axis=0)
    st = np.repeat(stratum, nh * ns * n_rings)
    hh = np.tile(np.repeat(heights, ns * n_rings), n)
    an = np.tile(np.arange(n_rings), n * nh * ns)
    return np.column_stack([X, st, hh, an]).astype(np.float32)


def target_vector(Y):
    """(n, h, ring, sector) -> (n·h·s·ring,), same order as `table`."""
    return np.transpose(Y, (0, 1, 3, 2)).reshape(-1)


def from_table(p, n, nh, na, ns):
    return np.transpose(p.reshape(n, nh, ns, na), (0, 1, 3, 2))


def pyramid_descriptors(hz, rows, cols):
    """B2 descriptors: (n, h, s, 2·b horizons + 2·b means), same layout as `table`."""
    hor, opa = hz(rows, cols)                                  # (n, h, 2, b, s), (n, 2, b, s)
    n, nh, ns = len(rows), hz.h.size, hz.ns
    hor = hor.transpose(0, 1, 4, 3, 2).reshape(n, nh, ns, -1)
    opa = np.repeat(opa.transpose(0, 3, 2, 1).reshape(n, ns, -1)[:, None], nh, axis=1)
    return np.concatenate([hor, opa], axis=-1)


def predict_b2(model, hz, rows, cols, stratum, heights, n_rings=2):
    """V predicted by B2: (n, h, ring, sector)."""
    D = pyramid_descriptors(hz, rows, cols)
    p = model.predict(table(D, stratum, heights, n_rings))
    return np.clip(from_table(p, len(rows), len(heights), n_rings, hz.ns), 0, 1)


# =============================================================================
# Step 2: training and evaluation
# =============================================================================
def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["surrogate"]["out_dir"]
    tr = np.load(out / "labels.npz")
    te = np.load(out / "test.npz")
    Y, st, sp = tr["Y"], tr["stratum"].astype(int), tr["split"]
    h = tr["heights"]
    _, nh, na, ns = Y.shape
    n_bin = bin_areas(cfg)
    Yt, stt = te["Y"], te["stratum"].astype(int)
    inside, xy = site_interior(cfg, ROOT, te["rows"], te["cols"])
    trn, val = sp == 0, sp == 1

    # --- B0
    mean = np.stack([Y[trn & (st == k)].mean(axis=0) for k in range(3)])
    P0 = mean[stt]
    P0v = mean[st[val]]

    # --- B1
    s = prepare_surfaces(cfg)
    t0 = time.time()
    D = descriptors(s, cfg, tr["rows"], tr["cols"])
    t_desc = time.time() - t0
    Dt = descriptors(s, cfg, te["rows"], te["cols"])
    X, y = table(D[trn], st[trn], h, na), target_vector(Y[trn])
    print(f"B1: {X.shape[0]:,} rows × {X.shape[1]} variables; descriptors "
          f"{1e3 * t_desc / len(Y):.2f} ms per position")
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                       categorical_features=[X.shape[1] - 3],
                                       random_state=cfg["project"]["random_seed"])
    t0 = time.time()
    gb.fit(X, y)
    t_fit = time.time() - t0
    t0 = time.time()
    P1 = np.clip(from_table(gb.predict(table(Dt, stt, h, na)), len(Yt), nh, na, ns), 0, 1)
    t_pred = time.time() - t0
    P1v = np.clip(from_table(gb.predict(table(D[val], st[val], h, na)), val.sum(), nh, na, ns),
                  0, 1)
    print(f"  training {t_fit:.0f} s; prediction {1e3 * t_pred / len(Yt):.2f} ms per position "
          f"(+ descriptors)")

    # --- B2: same rows, pyramid descriptors: horizons at each h + means
    from src.horizons import Horizons
    hz = Horizons(s, cfg)
    desc_pyr = lambda rows, cols: pyramid_descriptors(hz, rows, cols)
    Dp = desc_pyr(tr["rows"], tr["cols"])
    t0 = time.time()
    Dpt = desc_pyr(te["rows"], te["cols"])
    t_desc2 = (time.time() - t0) / len(Yt)
    gb2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                        categorical_features=[Dp.shape[-1]],
                                        random_state=cfg["project"]["random_seed"])
    gb2.fit(table(Dp[trn], st[trn], h, na), y)
    joblib.dump(gb2, out / "b2.joblib")                          # reused in 6.3
    t0 = time.time()
    P2 = np.clip(from_table(gb2.predict(table(Dpt, stt, h, na)), len(Yt), nh, na, ns), 0, 1)
    t_pred2 = (time.time() - t0) / len(Yt)
    print(f"B2: descriptors {1e3 * t_desc2:.3f} ms per position (test positions, not contiguous), "
          f"prediction {1e3 * t_pred2:.3f} ms")
    P2v = np.clip(from_table(gb2.predict(table(Dp[val], st[val], h, na)), val.sum(), nh, na, ns),
                  0, 1)

    res = pd.concat([
        report(P0v, Y[val], st[val], h, n_bin, name="B0 (validation)"),
        report(P1v, Y[val], st[val], h, n_bin, name="B1 (validation)"),
        report(P0, Yt, stt, h, n_bin, inside, xy, name="B0 (test)"),
        report(P1, Yt, stt, h, n_bin, inside, xy, name="B1 (test)"),
        report(P2v, Y[val], st[val], h, n_bin, name="B2 (validation)"),
        report(P2, Yt, stt, h, n_bin, inside, xy, name="B2 (test)"),
    ], ignore_index=True)
    res["cost_ms_position"] = np.where(res.model.str.startswith("B1"),
                                       1e3 * (t_desc / len(Y) + t_pred / len(Yt)), 0.0)
    # B2: pyramid descriptors + boosting prediction (dominated by the boosting)
    res.loc[res.model.str.startswith("B2"), "cost_ms_position"] = 1e3 * (t_desc2 + t_pred2)
    res.to_csv(out / "baselines.csv", index=False)
    np.savez_compressed(out / "baselines_test.npz", B0=P0.astype(np.float32),
                        B1=P1.astype(np.float32), B2=P2.astype(np.float32))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
