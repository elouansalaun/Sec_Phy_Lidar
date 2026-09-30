"""
src/validation_viewshed.py: validation of the LOS engine against gdal_viewshed.

Protocol : same surface (opaque DSM), same heights, no curvature.
           Only difference: exact R3 (ours) vs Wang, Robinson & White 2000 (GDAL).
Outputs  : comparison.csv (one row per pair) + agreement maps (GeoTIFF).

Usage    : python -m src.validation_viewshed
           python -m src.validation_viewshed diag C040 4   (diagnosis of one pair)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from numba import njit, prange
from scipy import ndimage

from src.rasters import load_grid, write_raster
from src.site import ROOT, Grid, load_config
from src.viewshed import optical_depth, read, to_pixel

OUT_OF_RANGE = 128


@njit(cache=True)
def los_prism(z_opaque, r0, c0, z_obs, r, c, z_t):
    """Exact visibility, surface made of prisms: traversal of every cell crossed by the
    segment (Amanatides & Woo 1987). Pixel centres on the integers."""
    dr, dc = r - r0, c - c0
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    td_r = abs(1.0 / dr) if dr != 0 else np.inf     # Δt to cross one row
    td_c = abs(1.0 / dc) if dc != 0 else np.inf     # Δt to cross one column
    tm_r, tm_c = 0.5 * td_r, 0.5 * td_c             # first edge: 0.5 px from the centre
    cr, cc, t_in = r0, c0, 0.0
    dz = z_t - z_obs
    while True:
        t_out = min(tm_r, tm_c, 1.0)
        if not (cr == r0 and cc == c0) and not (cr == r and cc == c):
            z_low = z_obs + dz * min(t_in, t_out) if dz >= 0 else z_obs + dz * max(t_in, t_out)
            if z_low < z_opaque[cr, cc]:
                return False
        if t_out >= 1.0:
            return True
        if tm_r < tm_c:
            cr += step_r; t_in = tm_r; tm_r += td_r
        elif tm_c < tm_r:
            cc += step_c; t_in = tm_c; tm_c += td_c
        else:                                        # exact passage through a corner
            cr += step_r; cc += step_c; t_in = tm_r; tm_r += td_r; tm_c += td_c


@njit(cache=True)
def prism_visible(z_opaque, z_tgt, r0, c0, z_obs, rows, cols):
    out = np.zeros(rows.size, np.bool_)
    for k in range(rows.size):
        out[k] = los_prism(z_opaque, r0, c0, z_obs, rows[k], cols[k], z_tgt[rows[k], cols[k]])
    return out

@njit(parallel=True, cache=True)
def blocking_margin(z_opaque, z_tgt, r0, c0, z_obs, radius_px, step, r_skip):
    """(margin, distance of the worst obstacle); margin = max(z_obstacle - z_ray) [m]."""
    H, W = z_opaque.shape
    out = np.full((H, W), np.nan, np.float32)
    where = np.full((H, W), np.nan, np.float32)
    for r in prange(max(r0 - radius_px, 0), min(r0 + radius_px + 1, H)):
        for c in range(max(c0 - radius_px, 0), min(c0 + radius_px + 1, W)):
            dr, dc = r - r0, c - c0
            d = np.sqrt(dr * dr + dc * dc)
            if d > radius_px or d == 0:
                continue
            dz = z_tgt[r, c] - z_obs
            marg, wd = -1e9, 0.0
            for i in range(1, int(d / step) + 1):
                s = i * step / d
                rr = int(np.floor(r0 + s * dr + 0.5))
                cc = int(np.floor(c0 + s * dc + 0.5))
                if rr == r and cc == c:
                    break
                if (rr == r0 and cc == c0) or i * step < r_skip:
                    continue
                m = z_opaque[rr, cc] - (z_obs + s * dz)
                if m > marg:
                    marg, wd = m, i * step
            out[r, c], where[r, c] = marg, wd
    return out, where


def diag(cand_id: str, h: float) -> None:
    cfg = load_config()
    v = cfg["viewshed"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    dsm_path = ROOT / cfg["rasters"]["out_dir"] / "dsm.tif"
    dsm = read(dsm_path)
    c = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id").loc[cand_id]
    r0, c0 = to_pixel(grid, c.geometry.x, c.geometry.y)
    oz, tz, R = c["z_support"] + h - dsm[r0, c0], v["target_height_m"], v["radius_m"]
    xc = grid.xmin + (c0 + 0.5) * grid.res
    yc = grid.ymax - (r0 + 0.5) * grid.res

    STEP_DIAG = 0.5
    args = (dsm, dsm + tz, r0, c0, dsm[r0, c0] + oz, int(R / grid.res), STEP_DIAG)
    marg, wd = blocking_margin(*args, 0.0)
    marg, wd = marg.astype(np.float64), wd.astype(np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        g = gdal_viewshed(dsm_path, xc, yc, oz, tz, R, Path(tmp) / "g.tif", grid)
    dom = ~np.isnan(marg) & (g != OUT_OF_RANGE)
    tau_o = ours_opaque(dsm, r0, c0, oz, tz, int(R / grid.res), grid.res)
    v_o, v_g = np.isfinite(tau_o), g == 255

    print(f"{cand_id} at {h:.0f} m: oz = {oz:.2f} m above the DSM")
    q = lambda x: np.round(np.percentile(x, [10, 50, 90]), 2) if x.size else "n/a"
    for name, m in [("GDAL only (we block)", dom & ~v_o & v_g),
                    ("ours only (GDAL blocks)", dom & v_o & ~v_g),
                    ("invisible for both", dom & ~v_o & ~v_g)]:
        x = marg[m]
        print(f"  {name:26s} n={x.size:>7,}  margin p10/50/90 = {q(x)} m")
    g_only = dom & ~v_o & v_g
    x = marg[g_only]
    if x.size:
        print(f"  GDAL only: {100 * np.mean(x < 0.5):.0f} % within 0.5 m, "
              f"{100 * np.mean(x < 1.0):.0f} % within 1 m")

    x, w = marg[g_only], wd[g_only]
    big = x > 1.0
    print(f"\n  GDAL only: distance of the worst obstacle to the mast p10/50/90 = {q(w)} m")
    if big.any():
        print(f"  GDAL only with margin > 1 m: {100 * big.mean():.0f} %; "
              f"of which obstacle < 3 m from the mast: {100 * np.mean(w[big] < 3):.0f} %")

    print(f"\n  IoU vs GDAL according to the tolerance ε [m]  ε=0     0.25    0.5     1.0")
    for r_skip in (0.0, 3.0):
        m_ = marg if r_skip == 0 else blocking_margin(*args, r_skip)[0].astype(np.float64)
        ious = []
        for eps in (0.0, 0.25, 0.5, 1.0):
            vo = m_ <= eps
            ious.append(np.sum(dom & vo & v_g) / max(np.sum(dom & (vo | v_g)), 1))
        print(f"  obstacles < {r_skip:.0f} m from the mast ignored      "
              + "  ".join(f"{i:.3f}" for i in ious))

    rng = np.random.default_rng(42)

    def referee(mask, n_max=20000):
        rr_, cc_ = np.nonzero(mask)
        if rr_.size > n_max:
            k = rng.choice(rr_.size, n_max, replace=False)
            rr_, cc_ = rr_[k], cc_[k]
        vis = prism_visible(dsm, dsm + tz, r0, c0, dsm[r0, c0] + oz,
                            rr_.astype(np.int64), cc_.astype(np.int64))
        return rr_, cc_, vis

    print("\n  Exact referee (prisms, Amanatides-Woo)")
    _, _, vis = referee(dom & ~v_o & v_g)
    print(f"    GDAL only : prism blocked (we are right)  {100 * (~vis).mean():5.1f} %  (n={vis.size:,})")
    _, _, vis = referee(dom & v_o & ~v_g)
    print(f"    ours only : prism visible (we are right)  {100 * vis.mean():5.1f} %  (n={vis.size:,})")
    rr_, cc_, vis = referee(dom & (v_o == v_g), 5000)
    print(f"    agreement : prism = ours                  {100 * (vis == v_o[rr_, cc_]).mean():5.1f} %  (n={vis.size:,})")


# =============================================================================
# Step 1: GDAL: run, then realign on the site grid
# =============================================================================
def gdal_viewshed(dsm_path: Path, x: float, y: float, oz: float, tz: float,
                  radius_m: float, out_path: Path, grid: Grid) -> np.ndarray:
    cmd = ["gdal_viewshed", "-ox", f"{x}", "-oy", f"{y}", "-oz", f"{oz}",
           "-tz", f"{tz}", "-md", f"{radius_m}", "-cc", "0",
           "-vv", "255", "-iv", "0", "-ov", f"{OUT_OF_RANGE}",
           str(dsm_path), str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)

    # GDAL crops its output to the max range: we paste it back into the full grid
    full = np.full((grid.height, grid.width), OUT_OF_RANGE, np.uint8)
    with rasterio.open(out_path) as src:
        a, t = src.read(1), src.transform
    r = int(round((grid.ymax - t.f) / grid.res))
    c = int(round((t.c - grid.xmin) / grid.res))
    full[r:r + a.shape[0], c:c + a.shape[1]] = a
    return full


# =============================================================================
# Step 2: our engine in "opaque DSM" mode
# =============================================================================
def ours_opaque(dsm: np.ndarray, r0: int, c0: int, oz: float, tz: float,
                radius_px: int, res: float) -> np.ndarray:
    return optical_depth(dsm, dsm, np.zeros_like(dsm), dsm + tz,
                         r0, c0, dsm[r0, c0] + oz, radius_px, res)


# =============================================================================
# Step 3: agreement metrics
# =============================================================================
def metrics(v_o: np.ndarray, v_g: np.ndarray, dom: np.ndarray, dist: np.ndarray) -> dict:
    a, b = v_o[dom], v_g[dom]
    n = float(a.size)
    tp, tn = float(np.sum(a & b)), float(np.sum(~a & ~b))
    only_o, only_g = float(np.sum(a & ~b)), float(np.sum(~a & b))
    po = (tp + tn) / n
    pe = ((tp + only_o) * (tp + only_g) + (tn + only_g) * (tn + only_o)) / n**2

    boundary = ndimage.binary_dilation(v_o) ^ ndimage.binary_erosion(v_o)
    d_edge = ndimage.distance_transform_edt(~boundary)       # distance to the boundary [px]
    dis = dom & (v_o != v_g)
    far = dom & (d_edge > 1)                          # outside a 1 px rim
    a2, b2 = v_o[far], v_g[far]
    iou_tol = float(np.sum(a2 & b2)) / max(float(np.sum(a2 | b2)), 1.0)
    return {
        "agreement": po,
        "kappa": (po - pe) / (1 - pe),
        "iou": tp / max(tp + only_o + only_g, 1.0),
        "ours_only": only_o / n,                               # visible for us, not for GDAL
        "gdal_only": only_g / n,
        "disagree_boundary_2px": float(np.mean(d_edge[dis] <= 2)) if dis.any() else 1.0,
        "disagree_beyond_200m": float(np.mean(dist[dis] > 200)) if dis.any() else 0.0,
        "iou_tolerant": iou_tol,
        "visible_share": (tp + only_o) / n,
    }


def agreement_map(v_o, v_g, dom) -> np.ndarray:
    """1 = visible (agreement), 2 = invisible (agreement), 3 = ours only, 4 = GDAL only."""
    m = np.zeros(v_o.shape, np.uint8)
    m[dom & v_o & v_g], m[dom & ~v_o & ~v_g] = 1, 2
    m[dom & v_o & ~v_g], m[dom & ~v_o & v_g] = 3, 4
    return m


# =============================================================================
# Step 4: one (position, height) pair
# =============================================================================
def run_case(c, h, dsm, dsm_path, grid, v, tmp: Path):
    r0, c0 = to_pixel(grid, c.geometry.x, c.geometry.y)
    oz = c["z_support"] + h - dsm[r0, c0]
    if oz <= 0:
        return None                                  # sensor below the DSM: outside GDAL
    xc = grid.xmin + (c0 + 0.5) * grid.res           # cell centre: no ambiguity
    yc = grid.ymax - (r0 + 0.5) * grid.res
    tz, R = v["target_height_m"], v["radius_m"]

    t0 = time.perf_counter()
    tau = ours_opaque(dsm, r0, c0, oz, tz, int(R / grid.res), grid.res)
    t_o = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = gdal_viewshed(dsm_path, xc, yc, oz, tz, R, tmp / f"{c['id']}_{h:.0f}.tif", grid)
    t_g = time.perf_counter() - t0

    rr, cc = np.indices(dsm.shape)
    dist = np.hypot(rr - r0, cc - c0) * grid.res
    dom = ~np.isnan(tau) & (g != OUT_OF_RANGE)
    v_o, v_g = np.isfinite(tau), g == 255
    m = {"id": c["id"], "support": c["support"], "h": h, "oz": oz,
         **metrics(v_o, v_g, dom, dist), "t_ours": t_o, "t_gdal": t_g}
    return m, agreement_map(v_o, v_g, dom)


# =============================================================================
# Step 5: every pair, summary, maps of the median and worst cases
# =============================================================================
def main() -> None:
    cfg = load_config()
    v = cfg["viewshed"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    dsm_path = ROOT / cfg["rasters"]["out_dir"] / "dsm.tif"
    dsm = read(dsm_path)
    assert not np.isnan(dsm).any(), "DSM with holes: GDAL would read -9999 as an elevation"
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    out = ROOT / cfg["validation"]["out_dir"] / "viewshed"
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for _, c in cand.iterrows():
            for h in map(float, c["heights_m"].split(",")):
                res = run_case(c, h, dsm, dsm_path, grid, v, Path(tmp))
                if res is None:
                    skipped += 1
                    continue
                rows.append(res[0])

        df = pd.DataFrame(rows)
        df.to_csv(out / "comparison.csv", index=False)

        for lab, i in (("median", (df["kappa"] - df["kappa"].median()).abs().idxmin()),
                       ("worst", df["kappa"].idxmin())):
            r = df.loc[i]
            c = cand.set_index("id").loc[r["id"]].copy()
            c["id"] = r["id"]
            _, amap = run_case(c, r["h"], dsm, dsm_path, grid, v, Path(tmp))
            write_raster(out / f"agreement_{lab}_{r['id']}_{r['h']:.0f}m.tif", amap, grid)

    cols = ["agreement", "kappa", "iou", "ours_only", "gdal_only",
            "disagree_boundary_2px", "disagree_beyond_200m"]
    print(f"{len(df)} pairs compared, {skipped} skipped (sensor below the DSM)\n")
    print(df[cols].describe(percentiles=[0.05, 0.5]).loc[["min", "5%", "50%", "mean"]].T.round(3))
    print("\nMedians per support:")
    print(df.groupby("support")[["kappa", "iou", "disagree_boundary_2px"]].median().round(3))
    print(f"\nMedian time: ours {1000 * df.t_ours.median():.0f} ms | "
          f"GDAL {1000 * df.t_gdal.median():.0f} ms (process + I/O included)")
    print(f"Results -> {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "diag":
        diag(sys.argv[2], float(sys.argv[3]))
    else:
        main()
