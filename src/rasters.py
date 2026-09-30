"""
src/rasters.py: base rasters (DTM, DSM, heights, vegetation) on the site grid.

Inputs  : LAZ tiles (data/raw), grid.yaml (produced by src.site).
Outputs : GeoTIFFs aligned on the grid in data/processed/rasters.

Method  : tiles are read in chunks (laspy), accumulated per pixel (numpy),
          then the products are built. Constant memory whatever the number of points.

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
# Step 1: grid and pixel indexing
# =============================================================================
def load_grid(path: Path) -> Grid:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Grid(crs=d["crs"], xmin=d["xmin"], ymin=d["ymin"],
                xmax=d["xmax"], ymax=d["ymax"], res=d["res"])


def pixel_index(x: np.ndarray, y: np.ndarray, grid: Grid):
    """Flat pixel index (row * width + column) and mask of the points inside the grid.

    Raster convention: row 0 at the top (ymax), column 0 on the left (xmin).
    A point on the left/top edge of a pixel belongs to that pixel.
    """
    col = np.floor((x - grid.xmin) / grid.res).astype(np.int64)
    row = np.floor((grid.ymax - y) / grid.res).astype(np.int64)
    inside = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    return row[inside] * grid.width + col[inside], inside


# =============================================================================
# Step 2: per-pixel statistics accumulator
# =============================================================================
@dataclass
class Accumulator:
    """Statistics of Z per pixel, updated chunk by chunk."""
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
# Step 3: reading the tiles in chunks
# =============================================================================
def tile_intersects(path: str, grid: Grid) -> bool:
    with laspy.open(path) as f:
        h = f.header
        return not (h.maxs[0] < grid.xmin or h.mins[0] > grid.xmax or
                    h.maxs[1] < grid.ymin or h.mins[1] > grid.ymax)


def accumulate(paths: list[str], grid: Grid, classes: dict[str, list[int]],
               chunk_size: int) -> tuple[dict[str, Accumulator], np.ndarray]:
    n = grid.width * grid.height
    groups = [g for g in classes if g != "noise"]
    acc = {g: Accumulator(n) for g in groups + ["surface"]}   # surface = everything but noise
    other_multi = np.zeros(n, np.int32)   # "other" points from pulses with multiple returns
    first_all = np.zeros(n, np.int32)     # first returns, noise excluded
    first_ground = np.zeros(n, np.int32)  # first returns classified as ground

    for path in paths:
        if not tile_intersects(path, grid):
            print(f"  {Path(path).name}: outside the grid, skipped")
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
                m = ~np.isin(cls, classes["noise"])
                acc["surface"].add(idx[m], z[m])
                m = np.isin(cls, classes["other"]) & (nret > 1)
                other_multi += np.bincount(idx[m], minlength=n).astype(np.int32)
                rnum = np.asarray(pts.return_number)[inside]
                first = (rnum == 1) & ~np.isin(cls, classes["noise"])
                first_all += np.bincount(idx[first], minlength=n).astype(np.int32)
                m = first & np.isin(cls, classes["ground"])
                first_ground += np.bincount(idx[m], minlength=n).astype(np.int32)
        print(f"  {Path(path).name}: {kept:,} points in the grid ({time.time() - t0:.0f} s)")
    return acc, other_multi, first_all, first_ground



# =============================================================================
# Step 4: building the products
# =============================================================================
def fill_holes(arr: np.ndarray, max_dist_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Fills empty pixels by interpolation (inverse distance). Returns (raster, filled_mask)."""
    valid = ~np.isnan(arr)
    filled = fillnodata(np.where(valid, arr, 0).astype(np.float32),
                        mask=valid.astype(np.uint8), max_search_distance=max_dist_px,
                        smoothing_iterations=0)
    filled = np.where(valid, arr, filled)
    still_empty = ~valid & (filled == 0)
    filled[still_empty] = np.nan
    return filled, ~valid & ~still_empty


def build_products(acc: dict[str, Accumulator], other_multi: np.ndarray,
                   first_all: np.ndarray, first_ground: np.ndarray,
                   grid: Grid, max_dist_px: int) -> dict[str, np.ndarray]:
    shape = (grid.height, grid.width)
    r = lambda a: a.reshape(shape)

    dtm, dtm_filled = fill_holes(r(acc["ground"].mean()), max_dist_px)
    dsm, dsm_filled = fill_holes(r(acc["surface"].max()), max_dist_px)
    dsm = np.fmax(dsm, dtm)                       # the surface is never below the ground

    building = r(acc["building"].count) > 0
    h_building = np.where(building, r(acc["building"].max()) - dtm, 0.0)
    other = r(acc["other"].count) > 0
    h_other = np.where(other, np.clip(r(acc["other"].max()) - dtm, 0, None), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_multi = np.where(other, r(other_multi) / r(acc["other"].count), np.nan)

    return {
        "dtm": dtm,
        "dsm": dsm,
        "dtm_filled": dtm_filled.astype(np.uint8),
        "building_mask": building.astype(np.uint8),
        "building_h": np.clip(h_building, 0, None),
        "other_h": h_other,
        "other_multi_frac": frac_multi,
        "density": r(acc["surface"].count).astype(np.float32) / grid.res**2,
        "gap_fraction": (r(first_ground) + 0.5) / (r(first_all) + 1.0),
        "n_first": r(first_all).astype(np.float32)
    }



# =============================================================================
# Step 5: GeoTIFF writing
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
# Step 6: orchestration and quality control
# =============================================================================
def main() -> None:
    cfg = load_config()
    rc = cfg["rasters"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    paths = sorted(p for p in glob.glob(str(ROOT / rc["tiles_glob"]))
                   if not p.lower().endswith(".copc.laz"))      # skips the QGIS indexes
    print(f"Grid {grid.width} x {grid.height} px at {grid.res} m, {len(paths)} tiles")

    acc, other_multi, first_all, first_ground = accumulate(paths, grid, rc["classes"],
                                                           rc["chunk_size"])
    products = build_products(acc, other_multi, first_all, first_ground, grid,
                              rc["fill_max_distance_px"])

    out_dir = ROOT / rc["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in products.items():
        write_raster(out_dir / f"{name}.tif", arr, grid)

    n = grid.width * grid.height
    print("\nQuality control")
    print(f"  pixels without measured ground (filled DTM): {100 * products['dtm_filled'].mean():.1f} %")
    print(f"  building pixels                            : {100 * products['building_mask'].mean():.1f} %")
    print(f"  DTM min / max                              : {np.nanmin(products['dtm']):.1f} / {np.nanmax(products['dtm']):.1f} m")
    print(f"  max building height                        : {np.nanmax(products['building_h']):.1f} m")
    print(f"  median density                             : {np.median(products['density']):.1f} pts/m²")
    print(f"  pixels still empty (DTM)                   : {int(np.isnan(products['dtm']).sum())} / {n}")
    gap, h_veg = products["gap_fraction"], products["other_h"]
    building, bare = products["building_mask"] == 1, (h_veg < 0.3) & (products["building_mask"] == 0)
    q = lambda a: np.round(np.percentile(a, [10, 50, 90]), 2)
    print(f"  gap fraction p10/50/90, bare ground        : {q(gap[bare])}")
    print(f"  gap fraction p10/50/90, vegetation         : {q(gap[h_veg > 2])}")
    print(f"  gap fraction p10/50/90, buildings          : {q(gap[building])}")
    print(f"  first returns / px (median)                : {np.median(products['n_first']):.0f}")
    print(f"Rasters written to {out_dir}")


if __name__ == "__main__":
    main()
