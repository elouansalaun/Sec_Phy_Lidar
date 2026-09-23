"""
src/viewshed.py — Épaisseur optique τ de la ligne de visée (moteur exact R3).

Modèle : terrain + bâtiments opaques ; végétation semi-transparente (Beer-Lambert),
         k = -ln(P_gap) / h_veg, P_gap mesurée sur les premiers retours LiDAR.
Sortie : τ par cellule. NaN = hors portée, inf = bloqué, T = exp(-τ) sinon.

Usage  : python -m src.viewshed C040 4
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from numba import njit, prange

from src.rasters import load_grid, write_raster
from src.site import ROOT, Grid, load_config


# =============================================================================
# Étape 1 — Surfaces : ce qui bloque, ce qui atténue, où est la cible
# =============================================================================
@dataclass
class Surfaces:
    z_opaque: np.ndarray   # altitude du dessus opaque (terrain + bâti)
    z_veg: np.ndarray      # altitude du sommet de la végétation
    k_ext: np.ndarray      # coefficient d'extinction [m⁻¹]
    z_tgt: np.ndarray      # altitude de la cible (MNT + h_cible)
    grid: Grid


def read(path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def prepare_surfaces(cfg: dict) -> Surfaces:
    v, d = cfg["viewshed"], ROOT / cfg["rasters"]["out_dir"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    mnt, bati_h, autre_h, gap = (read(d / f"{n}.tif")
                                 for n in ("mnt", "bati_h", "autre_h", "gap_fraction"))
    veg = autre_h > v["veg_min_h_m"]
    k = np.zeros_like(mnt)
    k[veg] = -np.log(gap[veg]) / np.maximum(autre_h[veg], 0.5)
    return Surfaces(z_opaque=mnt + bati_h,
                    z_veg=mnt + np.where(veg, autre_h, 0.0),
                    k_ext=k,
                    z_tgt=mnt + v["target_height_m"],
                    grid=grid)


# =============================================================================
# Étape 2 — Noyau : parcours exact des cellules traversées (Amanatides & Woo 1987)
#           Surface en prismes ; τ = Σ k · longueur 3D sous la canopée, par cellule
# =============================================================================
@njit(cache=True)
def ray_tau(z_opaque, z_veg, k_ext, r0, c0, z_obs, r, c, z_t, res):
    """τ le long du segment observateur -> cible ; inf si un prisme opaque est touché."""
    dr, dc = r - r0, c - c0
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    td_r = abs(1.0 / dr) if dr != 0 else np.inf        # Δt pour franchir une ligne
    td_c = abs(1.0 / dc) if dc != 0 else np.inf        # Δt pour franchir une colonne
    tm_r, tm_c = 0.5 * td_r, 0.5 * td_c                # premier bord à 0,5 px du centre
    dz = z_t - z_obs
    L = np.sqrt((dr * dr + dc * dc) * res * res + dz * dz)   # longueur 3D totale [m]
    cr, cc, t_in, tau = r0, c0, 0.0, 0.0
    while True:
        t_out = min(tm_r, tm_c, 1.0)
        if not (cr == r0 and cc == c0) and not (cr == r and cc == c):
            za, zb = z_obs + dz * t_in, z_obs + dz * t_out  # rayon à l'entrée / la sortie
            lo, hi = min(za, zb), max(za, zb)
            if lo < z_opaque[cr, cc]:
                return np.inf
            zv = z_veg[cr, cc]
            if lo < zv:                                      # passage (partiel) sous la canopée
                frac = 1.0 if hi <= zv else (zv - lo) / (hi - lo)
                tau += k_ext[cr, cc] * frac * (t_out - t_in) * L
        if t_out >= 1.0:
            return tau
        if tm_r < tm_c:
            cr += step_r; t_in = tm_r; tm_r += td_r
        elif tm_c < tm_r:
            cc += step_c; t_in = tm_c; tm_c += td_c
        else:                                                # passage exact par un coin
            cr += step_r; cc += step_c; t_in = tm_r; tm_r += td_r; tm_c += td_c


@njit(parallel=True, cache=True)
def optical_depth(z_opaque, z_veg, k_ext, z_tgt, r0, c0, z_obs, radius_px, res):
    """τ par cellule du disque. NaN = hors portée, inf = bloqué."""
    H, W = z_opaque.shape
    tau = np.full((H, W), np.nan, np.float32)
    for r in prange(max(r0 - radius_px, 0), min(r0 + radius_px + 1, H)):
        for c in range(max(c0 - radius_px, 0), min(c0 + radius_px + 1, W)):
            dr, dc = r - r0, c - c0
            zt = z_tgt[r, c]
            if dr * dr + dc * dc > radius_px * radius_px or np.isnan(zt):
                continue
            if dr == 0 and dc == 0:
                tau[r, c] = 0.0
                continue
            tau[r, c] = ray_tau(z_opaque, z_veg, k_ext, r0, c0, z_obs, r, c, zt, res)
    return tau


# =============================================================================
# Étape 3 — Observateur et appel
# =============================================================================
def to_pixel(grid: Grid, x: float, y: float) -> tuple[int, int]:
    return int((grid.ymax - y) // grid.res), int((x - grid.xmin) // grid.res)


def compute_tau(s: Surfaces, x: float, y: float, z_obs: float, cfg: dict) -> np.ndarray:
    v, g = cfg["viewshed"], s.grid
    r0, c0 = to_pixel(g, x, y)
    return optical_depth(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, r0, c0, z_obs,
                         int(v["radius_m"] / g.res), g.res)


# =============================================================================
# Étape 4 — Démo : un candidat, statistiques, GeoTIFF de transmittance
# =============================================================================
def main() -> None:
    cand_id, h = sys.argv[1], float(sys.argv[2])
    cfg = load_config()
    s = prepare_surfaces(cfg)
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id").loc[cand_id]
    z_obs = cand["z_support"] + h

    for run in ("compilation + calcul", "calcul seul"):
        t0 = time.time()
        tau = compute_tau(s, cand.geometry.x, cand.geometry.y, z_obs, cfg)
        print(f"  {run:22s}: {time.time() - t0:.2f} s")

    g = s.grid
    r0, c0 = to_pixel(g, cand.geometry.x, cand.geometry.y)
    rr, cc = np.indices(tau.shape)
    dist = np.hypot(rr - r0, cc - c0) * g.res
    sol = s.z_opaque < s.z_tgt              # cellule où un intrus peut se tenir (hors bâti)
    t64 = tau.astype(np.float64)            # évite le sous-dépassement de exp en float32

    print(f"\n{cand_id} ({cand['support']}) à {h:.0f} m — z_obs = {z_obs:.1f} m")
    print(f"  {'bande':>11} {'n sol':>8} {'opaque':>7} {'T<1%':>6} {'1-50%':>6} {'≥50%':>6}")
    for a, b in [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400)]:
        m = sol & (dist >= a) & (dist < b) & ~np.isnan(t64)
        t = t64[m]
        T = np.exp(-t)
        fin = np.isfinite(t)
        print(f"  {a:>4}-{b:<4} m {m.sum():>8,} {100 * (~fin).mean():>6.1f}% "
              f"{100 * (fin & (T < 0.01)).mean():>5.1f}% "
              f"{100 * ((T >= 0.01) & (T < 0.5)).mean():>5.1f}% {100 * (T >= 0.5).mean():>5.1f}%")

    out = ROOT / cfg["viewshed"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / f"T_{cand_id}_{h:.0f}m.tif", np.exp(-t64), s.grid)
    print(f"  -> {out / f'T_{cand_id}_{h:.0f}m.tif'}")


if __name__ == "__main__":
    main()