"""
src/coverage.py: coverage matrix of the MCLP.
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import sparse

from src.rasters import write_raster
from src.sensors import Geometry, load_sensors, p_camera, p_radar
from src.site import ROOT, load_config
from src.viewshed import compute_tau, prepare_surfaces, to_pixel


# =============================================================================
# Step 1: weighted cells: rasterized coverage zones, ground only
# =============================================================================
def weighted_cells(s, cover: gpd.GeoDataFrame):
    g = s.grid
    zone = rasterize(((geom, i + 1) for i, geom in enumerate(cover.geometry)),
                     out_shape=(g.height, g.width),
                     transform=from_origin(g.xmin, g.ymax, g.res, g.res),
                     fill=0, dtype="int32")                    # pixel-centre rule
    ground = s.z_opaque < s.z_tgt                                # an intruder can stand there
    rows, cols = np.nonzero((zone > 0) & ground)
    z = zone[rows, cols] - 1
    w = cover["weight"].to_numpy(float)[z] * g.res ** 2

    chk = pd.DataFrame({
        "zone": cover["source"], "weight": cover["weight"],
        "polygon_area_m2": cover.area.round(0),
        "rasterized_px": [int((zone == i + 1).sum()) for i in range(len(cover))],
        "kept_px": np.bincount(z, minlength=len(cover)),
    })
    return rows, cols, z, w, chk


# =============================================================================
# Step 2: geometry restricted to the cells (avoids computing the whole grid)
# =============================================================================
def cell_geometry(s, rows, cols, r0: int, c0: int, z_obs: float) -> Geometry:
    res = s.grid.res
    return Geometry(dx=(cols - c0) * res, dy=(r0 - rows) * res,
                    dz=z_obs - s.z_tgt[rows, cols],
                    h_rel=float(z_obs - s.z_tgt[r0, c0]))


# =============================================================================
# Step 3: building the matrix
# =============================================================================
def build_matrix(s, cand, rows, cols, w, cfg):
    cams, radars, azs, tilts = load_sensors(cfg)
    thr = cfg["coverage"]["p_min"]
    data, idx, indptr, meta = [], [], [0], []
    lost = total = 0.0
    n_empty = 0

    for _, c in cand.iterrows():
        r0, c0 = to_pixel(s.grid, c.geometry.x, c.geometry.y)
        for h in map(float, c["heights_m"].split(",")):
            z_obs = c["z_support"] + h
            tau = compute_tau(s, c.geometry.x, c.geometry.y, z_obs, cfg)[rows, cols]
            g = cell_geometry(s, rows, cols, r0, c0, z_obs)
            confs = [(cam.name, az, cam.tilt_deg(g.h_rel), cam.cost, p_camera(tau, g, cam, az))
                     for cam in cams for az in azs]
            confs += [(rad.name, np.nan, tl, rad.cost, p_radar(tau, g, rad, tl))
                      for rad in radars for tl in tilts]

            for name, az, tilt, cost, p in confs:
                keep = p >= thr
                total += float(w @ p)
                lost += float(w[~keep] @ p[~keep])
                if not keep.any():
                    n_empty += 1
                    continue
                nz = np.flatnonzero(keep)
                idx.append(nz)
                data.append(p[nz].astype(np.float32))
                indptr.append(indptr[-1] + nz.size)
                meta.append({"config": len(meta), "cand": c["id"], "support": c["support"],
                             "h_vegetation": c["h_vegetation"], "h": h,
                             "sensor": name, "azimuth": az, "tilt": tilt,
                             "support_cost": c["cost_base"] + c["cost_per_m"] * h,
                             "sensor_cost": cost,
                             "cov_w": float(w[nz] @ p[nz])})

    A = sparse.csr_matrix((np.concatenate(data), np.concatenate(idx), np.asarray(indptr)),
                          shape=(len(meta), rows.size))
    return A, pd.DataFrame(meta), n_empty, lost / total


# =============================================================================
# Step 4: orchestration, checks, diagnostics
# =============================================================================
def main() -> None:
    cfg = load_config()
    s = prepare_surfaces(cfg)
    cover = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="coverage")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    out = ROOT / cfg["coverage"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    rows, cols, z, w, chk = weighted_cells(s, cover)
    W = w.sum()
    print("Rasterization check of the zones:")
    print(chk.to_string(index=False))
    print(f"\n{rows.size:,} cells, total weight W = {W:,.0f}\n")

    t0 = time.time()
    A, df, n_empty, lost = build_matrix(s, cand, rows, cols, w, cfg)
    df["cov_frac"] = df["cov_w"] / W
    print(f"{len(df):,} useful configurations, {n_empty} discarded (no coverage) "
          f"in {time.time() - t0:.0f} s")
    print(f"A: {A.shape[0]:,} × {A.shape[1]:,}, {A.nnz:,} non-zeros "
          f"(density {A.nnz / np.prod(A.shape):.1%})")
    print(f"Loss due to the threshold p_min = {cfg['coverage']['p_min']}: "
          f"{100 * lost:.3f} % of the weighted coverage\n")

    print("Share of the weight covered by ONE configuration (Σ w·P / W):")
    print(df.groupby("sensor")["cov_frac"]
            .describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]].round(3))
    print("\nMedian by support × sensor:")
    print(df.pivot_table(index="support", columns="sensor", values="cov_frac",
                         aggfunc="median").round(3))
    print("\n10 best single configurations:")
    print(df.nlargest(10, "cov_frac")[["cand", "support", "h", "sensor", "azimuth", "tilt",
                                       "support_cost", "sensor_cost", "cov_frac"]]
            .round(3).to_string(index=False))

    # Best single configuration per cell: where is the site structurally poorly seen?
    p_max = np.asarray(A.max(axis=0).todense()).ravel()
    grid_p = np.full(s.z_tgt.shape, np.nan)
    grid_p[rows, cols] = p_max
    write_raster(out / "p_max.tif", grid_p, s.grid)
    weak = p_max < 0.5
    print(f"\nCells whose best configuration gives P < 0.5: "
          f"{100 * weak.mean():.1f} % of the cells, {100 * w[weak].sum() / W:.1f} % of the weight")
    print(pd.Series(p_max).groupby(cover["source"].to_numpy()[z])
            .agg(["median", lambda x: (x < 0.5).mean()])
            .set_axis(["median p_max", "share < 0.5"], axis=1).round(3))

    sparse.save_npz(out / "A.npz", A)
    np.savez(out / "cells.npz", rows=rows, cols=cols, zone=z, w=w)
    df.to_csv(out / "configs.csv", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
