"""
src/viewshield.py — Épaisseur optique τ de la ligne de visée (moteur exact R3).

Modèle : terrain + bâtiments opaques ; végétation semi-transparente (Beer-Lambert),
         k = -ln(P_gap) / h_veg, P_gap mesurée sur les premiers retours LiDAR.
Sortie : τ par cellule. NaN = hors portée, inf = bloqué, T = exp(-τ) sinon.

Usage  : python -m src.viewshield C040 4
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
    v, d = cfg["viewshield"], ROOT / cfg["rasters"]["out_dir"]
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
# Étape 2 — Noyau : un rayon exact vers chaque cellule du disque (R3)
# =============================================================================
@njit(parallel=True, cache=True)
def optical_depth(z_opaque, z_veg, k_ext, z_tgt, r0, c0, z_obs, radius_px, res, step):
    H, W = z_opaque.shape
    tau = np.full((H, W), np.nan, np.float32)            # NaN = hors portée
    for r in prange(max(r0 - radius_px, 0), min(r0 + radius_px + 1, H)):
        for c in range(max(c0 - radius_px, 0), min(c0 + radius_px + 1, W)):
            dr, dc = r - r0, c - c0
            d = np.sqrt(dr * dr + dc * dc)
            zt = z_tgt[r, c]
            if d > radius_px or np.isnan(zt):
                continue
            if d == 0:
                tau[r, c] = 0.0
                continue
            dz = zt - z_obs
            seg = step * res * np.sqrt(1.0 + (dz / (d * res)) ** 2)   # longueur 3D d'un pas [m]
            acc = 0.0
            blocked = False
            for i in range(1, int(d / step) + 1):
                s = i * step / d                                    # fraction du trajet
                rr = int(np.floor(r0 + s * dr + 0.5))
                cc = int(np.floor(c0 + s * dc + 0.5))
                if rr == r and cc == c:
                    break                                           # cible atteinte
                if rr == r0 and cc == c0:
                    continue                                        # pixel du mât lui-même
                z = z_obs + s * dz
                if z < z_opaque[rr, cc]:
                    blocked = True
                    break
                if z < z_veg[rr, cc]:
                    acc += k_ext[rr, cc] * seg
            tau[r, c] = np.inf if blocked else acc
    return tau


# =============================================================================
# Étape 3 — Observateur et appel
# =============================================================================
def to_pixel(grid: Grid, x: float, y: float) -> tuple[int, int]:
    return int((grid.ymax - y) // grid.res), int((x - grid.xmin) // grid.res)


def compute_tau(s: Surfaces, x: float, y: float, z_obs: float, cfg: dict) -> np.ndarray:
    v, g = cfg["viewshield"], s.grid
    r0, c0 = to_pixel(g, x, y)
    return optical_depth(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, r0, c0, z_obs,
                         int(v["radius_m"] / g.res), g.res, v["step_px"])


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

    out = ROOT / cfg["viewshield"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / f"T_{cand_id}_{h:.0f}m.tif", np.exp(-t64), s.grid)
    print(f"  -> {out / f'T_{cand_id}_{h:.0f}m.tif'}")


if __name__ == "__main__":
    main()