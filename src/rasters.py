"""
src/rasters.py — Rasters de base (MNT, MNS, hauteurs, végétation) sur la grille du site.

Entrées : tuiles LAZ (data/raw), grid.yaml (produit par src.site).
Sorties : GeoTIFF alignés sur la grille dans data/processed/rasters.

Méthode : lecture des tuiles par blocs (laspy), accumulation par pixel (numpy),
          puis construction des produits. Mémoire constante quel que soit le nombre de points.

Usage   : python -m src.rasters
"""
from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
import rasterio
import yaml
from rasterio.fill import fillnodata
from rasterio.transform import from_origin

from src.site import ROOT, Grid, load_config

NODATA = -9999.0



# =============================================================================
# Étape 1 — Grille et indexation des pixels
# =============================================================================
def load_grid(path: Path) -> Grid:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Grid(crs=d["crs"], xmin=d["xmin"], ymin=d["ymin"],
                xmax=d["xmax"], ymax=d["ymax"], res=d["res"])


def pixel_index(x: np.ndarray, y: np.ndarray, grid: Grid):
    """Indice de pixel à plat (ligne * largeur + colonne) et masque des points dans la grille.

    Convention raster : ligne 0 en haut (ymax), colonne 0 à gauche (xmin).
    Un point sur le bord gauche/haut d'un pixel appartient à ce pixel.
    """
    col = np.floor((x - grid.xmin) / grid.res).astype(np.int64)
    row = np.floor((grid.ymax - y) / grid.res).astype(np.int64)
    inside = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    return row[inside] * grid.width + col[inside], inside


# =============================================================================
# Étape 2 — Accumulateur de statistiques par pixel
# =============================================================================
@dataclass
class Accumulator:
    """Statistiques de Z par pixel, mises à jour bloc par bloc."""
    n_pixels: int
    count: np.ndarray = field(init=False)
    zsum: np.ndarray = field(init=False)
    zmin: np.ndarray = field(init=False)
    zmax: np.ndarray = field(init=False)

    def __post_init__(self):
        self.count = np.zeros(self.n_pixels, np.int32)
        self.zsum = np.zeros(self.n_pixels, np.float64)
        self.zmin = np.full(self.n_pixels, np.inf)
        self.zmax = np.full(self.n_pixels, -np.inf)

    def add(self, idx: np.ndarray, z: np.ndarray) -> None:
        if idx.size == 0:
            return
        self.count += np.bincount(idx, minlength=self.n_pixels).astype(np.int32)
        self.zsum += np.bincount(idx, weights=z, minlength=self.n_pixels)
        np.minimum.at(self.zmin, idx, z)
        np.maximum.at(self.zmax, idx, z)

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.zsum / self.count, np.nan)

    def max(self) -> np.ndarray:
        return np.where(self.count > 0, self.zmax, np.nan)




# =============================================================================
# Étape 3 — Lecture des tuiles par blocs
# =============================================================================
def tile_intersects(path: str, grid: Grid) -> bool:
    with laspy.open(path) as f:
        h = f.header
        return not (h.maxs[0] < grid.xmin or h.mins[0] > grid.xmax or
                    h.maxs[1] < grid.ymin or h.mins[1] > grid.ymax)


def accumulate(paths: list[str], grid: Grid, classes: dict[str, list[int]],
               chunk_size: int) -> tuple[dict[str, Accumulator], np.ndarray]:
    n = grid.width * grid.height
    groups = [g for g in classes if g != "bruit"]
    acc = {g: Accumulator(n) for g in groups + ["surface"]}   # surface = tout sauf bruit
    autre_multi = np.zeros(n, np.int32)   # points "autre" issus d'impulsions à retours multiples
    first_all = np.zeros(n, np.int32)   # premiers retours, hors bruit
    first_sol = np.zeros(n, np.int32)   # premiers retours classés sol

    for path in paths:
        if not tile_intersects(path, grid):
            print(f"  {Path(path).name} : hors grille, ignorée")
            continue
        t0, kept = time.time(), 0
        with laspy.open(path) as f:
            for pts in f.chunk_iterator(chunk_size):
                idx, inside = pixel_index(np.asarray(pts.x), np.asarray(pts.y), grid)
                z = np.asarray(pts.z)[inside]
                cls = np.asarray(pts.classification)[inside]
                nret = np.asarray(pts.number_of_returns)[inside]
                kept += idx.size

                for g in groups:
                    m = np.isin(cls, classes[g])
                    acc[g].add(idx[m], z[m])
                m = ~np.isin(cls, classes["bruit"])
                acc["surface"].add(idx[m], z[m])
                m = np.isin(cls, classes["autre"]) & (nret > 1)
                autre_multi += np.bincount(idx[m], minlength=n).astype(np.int32)
                rnum = np.asarray(pts.return_number)[inside]
                first = (rnum == 1) & ~np.isin(cls, classes["bruit"])
                first_all += np.bincount(idx[first], minlength=n).astype(np.int32)
                m = first & np.isin(cls, classes["sol"])
                first_sol += np.bincount(idx[m], minlength=n).astype(np.int32)
        print(f"  {Path(path).name} : {kept:,} points dans la grille ({time.time() - t0:.0f} s)")
    return acc, autre_multi, first_all, first_sol



# =============================================================================
# Étape 4 — Construction des produits
# =============================================================================
def fill_holes(arr: np.ndarray, max_dist_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Comble les pixels vides par interpolation (inverse distance). Retourne (raster, masque_comblé)."""
    valid = ~np.isnan(arr)
    filled = fillnodata(np.where(valid, arr, 0).astype(np.float32),
                        mask=valid.astype(np.uint8), max_search_distance=max_dist_px,
                        smoothing_iterations=0)
    filled = np.where(valid, arr, filled)
    still_empty = ~valid & (filled == 0)
    filled[still_empty] = np.nan
    return filled, ~valid & ~still_empty


def build_products(acc: dict[str, Accumulator], autre_multi: np.ndarray,
                   first_all: np.ndarray, first_sol: np.ndarray,
                   grid: Grid, max_dist_px: int) -> dict[str, np.ndarray]:
    shape = (grid.height, grid.width)
    r = lambda a: a.reshape(shape)

    mnt, mnt_filled = fill_holes(r(acc["sol"].mean()), max_dist_px)
    mns, mns_filled = fill_holes(r(acc["surface"].max()), max_dist_px)
    mns = np.fmax(mns, mnt)                       # la surface n'est jamais sous le sol

    bati = r(acc["batiment"].count) > 0
    h_bati = np.where(bati, r(acc["batiment"].max()) - mnt, 0.0)
    autre = r(acc["autre"].count) > 0
    h_autre = np.where(autre, np.clip(r(acc["autre"].max()) - mnt, 0, None), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_multi = np.where(autre, r(autre_multi) / r(acc["autre"].count), np.nan)

    return {
        "mnt": mnt,
        "mns": mns,
        "mnt_comble": mnt_filled.astype(np.uint8),
        "bati_mask": bati.astype(np.uint8),
        "bati_h": np.clip(h_bati, 0, None),
        "autre_h": h_autre,
        "autre_frac_multi": frac_multi,
        "densite": r(acc["surface"].count).astype(np.float32) / grid.res**2,
        "gap_fraction": (r(first_sol) + 0.5) / (r(first_all) + 1.0),
        "n_premiers": r(first_all).astype(np.float32)
    }



# =============================================================================
# Étape 5 — Écriture GeoTIFF
# =============================================================================
def write_raster(path: Path, arr: np.ndarray, grid: Grid) -> None:
    is_mask = arr.dtype == np.uint8
    profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=1,
                   dtype="uint8" if is_mask else "float32", crs=grid.crs,
                   transform=from_origin(grid.xmin, grid.ymax, grid.res, grid.res),
                   nodata=None if is_mask else NODATA,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    data = arr if is_mask else np.where(np.isnan(arr), NODATA, arr).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)



# =============================================================================
# Étape 6 — Orchestration et contrôle qualité
# =============================================================================
def main() -> None:
    cfg = load_config()
    rc = cfg["rasters"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    paths = sorted(p for p in glob.glob(str(ROOT / rc["tiles_glob"]))
                   if not p.lower().endswith(".copc.laz"))      # ignore les index QGIS
    print(f"Grille {grid.width} x {grid.height} px à {grid.res} m, {len(paths)} tuiles")

    acc, autre_multi, first_all, first_sol = accumulate(paths, grid, rc["classes"], rc["chunk_size"])
    products = build_products(acc, autre_multi, first_all, first_sol, grid, rc["fill_max_distance_px"])

    out_dir = ROOT / rc["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in products.items():
        write_raster(out_dir / f"{name}.tif", arr, grid)

    n = grid.width * grid.height
    print("\nContrôle qualité")
    print(f"  pixels sans sol mesuré (MNT comblé) : {100 * products['mnt_comble'].mean():.1f} %")
    print(f"  pixels bâtiment                     : {100 * products['bati_mask'].mean():.1f} %")
    print(f"  MNT min / max                       : {np.nanmin(products['mnt']):.1f} / {np.nanmax(products['mnt']):.1f} m")
    print(f"  hauteur bâti max                    : {np.nanmax(products['bati_h']):.1f} m")
    print(f"  densité médiane                     : {np.median(products['densite']):.1f} pts/m²")
    print(f"  pixels encore vides (MNT)           : {int(np.isnan(products['mnt']).sum())} / {n}")
    gap, h_veg = products["gap_fraction"], products["autre_h"]
    bati, sol_nu = products["bati_mask"] == 1, (h_veg < 0.3) & (products["bati_mask"] == 0)
    q = lambda a: np.round(np.percentile(a, [10, 50, 90]), 2)
    print(f"  gap fraction p10/50/90 — sol nu     : {q(gap[sol_nu])}")
    print(f"  gap fraction p10/50/90 — végétation : {q(gap[h_veg > 2])}")
    print(f"  gap fraction p10/50/90 — bâtiments  : {q(gap[bati])}")
    print(f"  premiers retours / px (médiane)     : {np.median(products['n_premiers']):.0f}")
    print(f"Rasters écrits dans {out_dir}")


if __name__ == "__main__":
    main()