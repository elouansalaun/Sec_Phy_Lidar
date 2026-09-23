"""
src/coverage.py — Matrice de couverture du MCLP.

Configuration i = (position, hauteur, capteur, azimut) ; cellule j = pixel de sol d'une zone
de couverture pondérée. A[i, j] = P_det de i sur j (creuse : P < p_min -> 0, perte mesurée).

Sorties : A.npz (CSR), configs.csv, cellules.npz, p_max.tif
Usage   : python -m src.coverage
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
# Étape 1 — Cellules pondérées : zones de couverture rastérisées, sol uniquement
# =============================================================================
def weighted_cells(s, cover: gpd.GeoDataFrame):
    g = s.grid
    zone = rasterize(((geom, i + 1) for i, geom in enumerate(cover.geometry)),
                     out_shape=(g.height, g.width),
                     transform=from_origin(g.xmin, g.ymax, g.res, g.res),
                     fill=0, dtype="int32")                    # règle du centre de pixel
    sol = s.z_opaque < s.z_tgt                                   # un intrus peut s'y tenir
    rows, cols = np.nonzero((zone > 0) & sol)
    z = zone[rows, cols] - 1
    w = cover["poids"].to_numpy(float)[z] * g.res ** 2

    chk = pd.DataFrame({
        "zone": cover["source"], "poids": cover["poids"],
        "surface_poly_m2": cover.area.round(0),
        "px_rasterises": [int((zone == i + 1).sum()) for i in range(len(cover))],
        "px_retenus": np.bincount(z, minlength=len(cover)),
    })
    return rows, cols, z, w, chk


# =============================================================================
# Étape 2 — Géométrie restreinte aux cellules (évite de calculer toute la grille)
# =============================================================================
def cell_geometry(s, rows, cols, r0: int, c0: int, z_obs: float) -> Geometry:
    res = s.grid.res
    return Geometry(dx=(cols - c0) * res, dy=(r0 - rows) * res,
                    dz=z_obs - s.z_tgt[rows, cols],
                    h_rel=float(z_obs - s.z_tgt[r0, c0]))


# =============================================================================
# Étape 3 — Construction de la matrice
# =============================================================================
def build_matrix(s, cand, rows, cols, w, cfg):
    cams, radars, azs, tilts = load_sensors(cfg)
    thr = cfg["coverage"]["p_min"]
    data, idx, indptr, meta = [], [], [0], []
    lost = total = 0.0
    n_empty = 0

    for _, c in cand.iterrows():
        r0, c0 = to_pixel(s.grid, c.geometry.x, c.geometry.y)
        for h in map(float, c["hauteurs_m"].split(",")):
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
                             "capteur": name, "azimut": az, "inclinaison": tilt,
                             "cout_support": c["cout_base"] + c["cout_par_m"] * h,
                             "cout_capteur": cost,
                             "couv_w": float(w[nz] @ p[nz])})

    A = sparse.csr_matrix((np.concatenate(data), np.concatenate(idx), np.asarray(indptr)),
                          shape=(len(meta), rows.size))
    return A, pd.DataFrame(meta), n_empty, lost / total


# =============================================================================
# Étape 4 — Orchestration, contrôles, diagnostics
# =============================================================================
def main() -> None:
    cfg = load_config()
    s = prepare_surfaces(cfg)
    cover = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="couverture")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    out = ROOT / cfg["coverage"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    rows, cols, z, w, chk = weighted_cells(s, cover)
    W = w.sum()
    print("Contrôle de rastérisation des zones :")
    print(chk.to_string(index=False))
    print(f"\n{rows.size:,} cellules, poids total W = {W:,.0f}\n")

    t0 = time.time()
    A, df, n_empty, lost = build_matrix(s, cand, rows, cols, w, cfg)
    df["couv_frac"] = df["couv_w"] / W
    print(f"{len(df):,} configurations utiles, {n_empty} écartées (aucune couverture) "
          f"— {time.time() - t0:.0f} s")
    print(f"A : {A.shape[0]:,} × {A.shape[1]:,}, {A.nnz:,} non nuls "
          f"(densité {A.nnz / np.prod(A.shape):.1%})")
    print(f"Perte due au seuil p_min = {cfg['coverage']['p_min']} : "
          f"{100 * lost:.3f} % de la couverture pondérée\n")

    print("Part du poids couverte par UNE configuration (Σ w·P / W) :")
    print(df.groupby("capteur")["couv_frac"]
            .describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]].round(3))
    print("\nMédiane par support × capteur :")
    print(df.pivot_table(index="support", columns="capteur", values="couv_frac",
                         aggfunc="median").round(3))
    print("\n10 meilleures configurations isolées :")
    print(df.nlargest(10, "couv_frac")[["cand", "support", "h", "capteur", "azimut", "inclinaison",
                                        "cout_support", "cout_capteur", "couv_frac"]]
            .round(3).to_string(index=False))

    # Meilleure configuration unique par cellule : où le site est-il structurellement mal vu ?
    p_max = np.asarray(A.max(axis=0).todense()).ravel()
    grid_p = np.full(s.z_tgt.shape, np.nan)
    grid_p[rows, cols] = p_max
    write_raster(out / "p_max.tif", grid_p, s.grid)
    faible = p_max < 0.5
    print(f"\nCellules dont la meilleure configuration donne P < 0,5 : "
          f"{100 * faible.mean():.1f} % des cellules, {100 * w[faible].sum() / W:.1f} % du poids")
    print(pd.Series(p_max).groupby(cover["source"].to_numpy()[z])
            .agg(["median", lambda x: (x < 0.5).mean()])
            .set_axis(["p_max médiane", "part < 0,5"], axis=1).round(3))

    sparse.save_npz(out / "A.npz", A)
    np.savez(out / "cellules.npz", rows=rows, cols=cols, zone=z, w=w)
    df.to_csv(out / "configs.csv", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()