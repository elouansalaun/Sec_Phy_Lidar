"""
src/validation_viewshed.py — Validation du moteur LOS contre gdal_viewshed.

Protocole : même surface (MNS opaque), mêmes hauteurs, pas de courbure.
            Seule différence : R3 exact (nous) vs Wang, Robinson & White 2000 (GDAL).
Sorties   : comparaison.csv (une ligne par couple) + cartes d'accord (GeoTIFF).

Usage     : python -m src.validation_viewshed
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from src.rasters import load_grid, write_raster
from src.site import ROOT, Grid, load_config
from src.viewshed import optical_depth, read, to_pixel

OUT_OF_RANGE = 128

from numba import njit, prange
import sys


@njit(cache=True)
def los_prism(z_opaque, r0, c0, z_obs, r, c, z_t):
    """Visibilité exacte, surface en prismes : parcours de toutes les cellules
    traversées par le segment (Amanatides & Woo 1987). Centres de pixels aux entiers."""
    dr, dc = r - r0, c - c0
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    td_r = abs(1.0 / dr) if dr != 0 else np.inf     # Δt pour franchir une ligne
    td_c = abs(1.0 / dc) if dc != 0 else np.inf     # Δt pour franchir une colonne
    tm_r, tm_c = 0.5 * td_r, 0.5 * td_c             # premier bord : à 0,5 px du centre
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
        else:                                        # passage exact par un coin
            cr += step_r; cc += step_c; t_in = tm_r; tm_r += td_r; tm_c += td_c


@njit(cache=True)
def prism_visible(z_opaque, z_tgt, r0, c0, z_obs, rows, cols):
    out = np.zeros(rows.size, np.bool_)
    for k in range(rows.size):
        out[k] = los_prism(z_opaque, r0, c0, z_obs, rows[k], cols[k], z_tgt[rows[k], cols[k]])
    return out

@njit(parallel=True, cache=True)
def blocking_margin(z_opaque, z_tgt, r0, c0, z_obs, radius_px, step, r_skip):
    """(marge, distance du pire obstacle) ; marge = max(z_obstacle - z_rayon) [m]."""
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
    mns_path = ROOT / cfg["rasters"]["out_dir"] / "mns.tif"
    mns = read(mns_path)
    c = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id").loc[cand_id]
    r0, c0 = to_pixel(grid, c.geometry.x, c.geometry.y)
    oz, tz, R = c["z_support"] + h - mns[r0, c0], v["target_height_m"], v["radius_m"]
    xc = grid.xmin + (c0 + 0.5) * grid.res
    yc = grid.ymax - (r0 + 0.5) * grid.res

    STEP_DIAG = 0.5
    args = (mns, mns + tz, r0, c0, mns[r0, c0] + oz, int(R / grid.res), STEP_DIAG)
    marg, wd = blocking_margin(*args, 0.0)
    marg, wd = marg.astype(np.float64), wd.astype(np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        g = gdal_viewshed(mns_path, xc, yc, oz, tz, R, Path(tmp) / "g.tif", grid)
    dom = ~np.isnan(marg) & (g != OUT_OF_RANGE)
    tau_o = ours_opaque(mns, r0, c0, oz, tz, int(R / grid.res), grid.res)
    v_o, v_g = np.isfinite(tau_o), g == 255

    print(f"{cand_id} à {h:.0f} m — oz = {oz:.2f} m au-dessus du MNS")
    q = lambda x: np.round(np.percentile(x, [10, 50, 90]), 2) if x.size else "—"
    for nom, m in [("GDAL seul (nous bloqués)", dom & ~v_o & v_g),
                   ("nous seuls (GDAL bloqué)", dom & v_o & ~v_g),
                   ("invisible des deux côtés", dom & ~v_o & ~v_g)]:
        x = marg[m]
        print(f"  {nom:26s} n={x.size:>7,}  marge p10/50/90 = {q(x)} m")
    x = marg[dom & ~v_o & v_g]
    if x.size:
        print(f"  GDAL seul : {100 * np.mean(x < 0.5):.0f} % à moins de 0,5 m, "
              f"{100 * np.mean(x < 1.0):.0f} % à moins de 1 m")

        g_only = dom & ~v_o & v_g
    x, w = marg[g_only], wd[g_only]
    big = x > 1.0
    print(f"\n  GDAL seul — distance du pire obstacle au mât p10/50/90 = {q(w)} m")
    if big.any():
        print(f"  GDAL seul avec marge > 1 m : {100 * big.mean():.0f} % ; "
              f"dont obstacle à < 3 m du mât : {100 * np.mean(w[big] < 3):.0f} %")

    print(f"\n  IoU vs GDAL selon la tolérance ε [m]        ε=0     0.25    0.5     1.0")
    for r_skip in (0.0, 3.0):
        m_ = marg if r_skip == 0 else blocking_margin(*args, r_skip)[0].astype(np.float64)
        ious = []
        for eps in (0.0, 0.25, 0.5, 1.0):
            vo = m_ <= eps
            ious.append(np.sum(dom & vo & v_g) / max(np.sum(dom & (vo | v_g)), 1))
        print(f"  obstacles à < {r_skip:.0f} m du mât ignorés      "
              + "  ".join(f"{i:.3f}" for i in ious))

        rng = np.random.default_rng(42)

    def arbitre(mask, n_max=20000):
        rr_, cc_ = np.nonzero(mask)
        if rr_.size > n_max:
            k = rng.choice(rr_.size, n_max, replace=False)
            rr_, cc_ = rr_[k], cc_[k]
        vis = prism_visible(mns, mns + tz, r0, c0, mns[r0, c0] + oz,
                            rr_.astype(np.int64), cc_.astype(np.int64))
        return rr_, cc_, vis

    print("\n  Arbitre exact (prismes, Amanatides-Woo)")
    _, _, vis = arbitre(dom & ~v_o & v_g)
    print(f"    GDAL seul  : prisme bloqué (donne raison à nous) {100 * (~vis).mean():5.1f} %  (n={vis.size:,})")
    _, _, vis = arbitre(dom & v_o & ~v_g)
    print(f"    nous seuls : prisme visible (donne raison à nous) {100 * vis.mean():5.1f} %  (n={vis.size:,})")
    rr_, cc_, vis = arbitre(dom & (v_o == v_g), 5000)
    print(f"    accord     : prisme = nous                        {100 * (vis == v_o[rr_, cc_]).mean():5.1f} %  (n={vis.size:,})")

        
# =============================================================================
# Étape 1 — GDAL : lancer, puis recaler sur la grille du site
# =============================================================================
def gdal_viewshed(mns_path: Path, x: float, y: float, oz: float, tz: float,
                  radius_m: float, out_path: Path, grid: Grid) -> np.ndarray:
    cmd = ["gdal_viewshed", "-ox", f"{x}", "-oy", f"{y}", "-oz", f"{oz}",
           "-tz", f"{tz}", "-md", f"{radius_m}", "-cc", "0",
           "-vv", "255", "-iv", "0", "-ov", f"{OUT_OF_RANGE}",
           str(mns_path), str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)

    # GDAL rogne sa sortie à la portée max : on la recolle dans la grille complète
    full = np.full((grid.height, grid.width), OUT_OF_RANGE, np.uint8)
    with rasterio.open(out_path) as src:
        a, t = src.read(1), src.transform
    r = int(round((grid.ymax - t.f) / grid.res))
    c = int(round((t.c - grid.xmin) / grid.res))
    full[r:r + a.shape[0], c:c + a.shape[1]] = a
    return full


# =============================================================================
# Étape 2 — Notre moteur en mode « MNS opaque »
# =============================================================================
def ours_opaque(mns: np.ndarray, r0: int, c0: int, oz: float, tz: float,
                radius_px: int, res: float) -> np.ndarray:
    return optical_depth(mns, mns, np.zeros_like(mns), mns + tz,
                         r0, c0, mns[r0, c0] + oz, radius_px, res)


# =============================================================================
# Étape 3 — Métriques d'accord
# =============================================================================
def metrics(v_o: np.ndarray, v_g: np.ndarray, dom: np.ndarray, dist: np.ndarray) -> dict:
    a, b = v_o[dom], v_g[dom]
    n = float(a.size)
    tp, tn = float(np.sum(a & b)), float(np.sum(~a & ~b))
    only_o, only_g = float(np.sum(a & ~b)), float(np.sum(~a & b))
    po = (tp + tn) / n
    pe = ((tp + only_o) * (tp + only_g) + (tn + only_g) * (tn + only_o)) / n**2

    boundary = ndimage.binary_dilation(v_o) ^ ndimage.binary_erosion(v_o)
    d_edge = ndimage.distance_transform_edt(~boundary)       # distance à la frontière [px]
    dis = dom & (v_o != v_g)
    far = dom & (d_edge > 1)                          # hors liseré de 1 px
    a2, b2 = v_o[far], v_g[far]
    iou_tol = float(np.sum(a2 & b2)) / max(float(np.sum(a2 | b2)), 1.0)
    return {
        "accord": po,
        "kappa": (po - pe) / (1 - pe),
        "iou": tp / max(tp + only_o + only_g, 1.0),
        "nous_seul": only_o / n,                               # visible pour nous, pas GDAL
        "gdal_seul": only_g / n,
        "desacc_frontiere_2px": float(np.mean(d_edge[dis] <= 2)) if dis.any() else 1.0,
        "desacc_au_dela_200m": float(np.mean(dist[dis] > 200)) if dis.any() else 0.0,
        "iou_tolerant": iou_tol,
        "part_visible": (tp + only_o) / n,
    }


def agreement_map(v_o, v_g, dom) -> np.ndarray:
    """1 = visible (accord), 2 = invisible (accord), 3 = nous seuls, 4 = GDAL seul."""
    m = np.zeros(v_o.shape, np.uint8)
    m[dom & v_o & v_g], m[dom & ~v_o & ~v_g] = 1, 2
    m[dom & v_o & ~v_g], m[dom & ~v_o & v_g] = 3, 4
    return m


# =============================================================================
# Étape 4 — Un couple (position, hauteur)
# =============================================================================
def run_case(c, h, mns, mns_path, grid, v, tmp: Path):
    r0, c0 = to_pixel(grid, c.geometry.x, c.geometry.y)
    oz = c["z_support"] + h - mns[r0, c0]
    if oz <= 0:
        return None                                  # capteur sous le MNS : hors GDAL
    xc = grid.xmin + (c0 + 0.5) * grid.res           # centre de cellule : aucune ambiguïté
    yc = grid.ymax - (r0 + 0.5) * grid.res
    tz, R = v["target_height_m"], v["radius_m"]

    t0 = time.perf_counter()
    tau = ours_opaque(mns, r0, c0, oz, tz, int(R / grid.res), grid.res)
    t_o = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = gdal_viewshed(mns_path, xc, yc, oz, tz, R, tmp / f"{c['id']}_{h:.0f}.tif", grid)
    t_g = time.perf_counter() - t0

    rr, cc = np.indices(mns.shape)
    dist = np.hypot(rr - r0, cc - c0) * grid.res
    dom = ~np.isnan(tau) & (g != OUT_OF_RANGE)
    v_o, v_g = np.isfinite(tau), g == 255
    m = {"id": c["id"], "support": c["support"], "h": h, "oz": oz,
         **metrics(v_o, v_g, dom, dist), "t_nous": t_o, "t_gdal": t_g}
    return m, agreement_map(v_o, v_g, dom)


# =============================================================================
# Étape 5 — Tous les couples, synthèse, cartes des cas médian et pire
# =============================================================================
def main() -> None:
    cfg = load_config()
    v = cfg["viewshed"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    mns_path = ROOT / cfg["rasters"]["out_dir"] / "mns.tif"
    mns = read(mns_path)
    assert not np.isnan(mns).any(), "MNS troué : GDAL lirait -9999 comme une altitude"
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    out = ROOT / cfg["validation"]["out_dir"] / "viewshed"
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for _, c in cand.iterrows():
            for h in map(float, c["hauteurs_m"].split(",")):
                res = run_case(c, h, mns, mns_path, grid, v, Path(tmp))
                if res is None:
                    skipped += 1
                    continue
                rows.append(res[0])

        df = pd.DataFrame(rows)
        df.to_csv(out / "comparaison.csv", index=False)

        for label, i in (("median", (df["kappa"] - df["kappa"].median()).abs().idxmin()),
                         ("pire", df["kappa"].idxmin())):
            r = df.loc[i]
            c = cand.set_index("id").loc[r["id"]].copy()
            c["id"] = r["id"]
            _, amap = run_case(c, r["h"], mns, mns_path, grid, v, Path(tmp))
            write_raster(out / f"accord_{label}_{r['id']}_{r['h']:.0f}m.tif", amap, grid)

    cols = ["accord", "kappa", "iou", "nous_seul", "gdal_seul",
            "desacc_frontiere_2px", "desacc_au_dela_200m"]
    print(f"{len(df)} couples comparés, {skipped} ignorés (capteur sous le MNS)\n")
    print(df[cols].describe(percentiles=[0.05, 0.5]).loc[["min", "5%", "50%", "mean"]].T.round(3))
    print("\nMédianes par support :")
    print(df.groupby("support")[["kappa", "iou", "desacc_frontiere_2px"]].median().round(3))
    print(f"\nTemps médian : nous {1000 * df.t_nous.median():.0f} ms | "
          f"GDAL {1000 * df.t_gdal.median():.0f} ms (processus + E/S inclus)")
    print(f"Résultats -> {out}")





if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "diag":
        diag(sys.argv[2], float(sys.argv[3]))
    else:
        main()