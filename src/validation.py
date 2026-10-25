"""
src/validation.py: validation of the DTM against two independent references.

1. PDAL (writers.gdal, binmode) on a test area: checks OUR gridding.
2. MRNF DTM (official product, other projection): checks the whole chain.

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.rasters import NODATA, load_grid, tile_intersects
from src.site import ROOT, Grid, load_config


# =============================================================================
# Step 1: test sub-grid, aligned on the main grid
# =============================================================================
def subgrid(grid: Grid, bounds, margin: float) -> Grid:
    """Sub-grid around `bounds`, whose edges fall on those of `grid`."""
    r = grid.res
    snap = lambda v, f: float(f((v - grid.xmin) / r) * r + grid.xmin)
    snapy = lambda v, f: float(f((v - grid.ymin) / r) * r + grid.ymin)
    xmin, ymin, xmax, ymax = bounds
    return Grid(crs=grid.crs,
                xmin=snap(xmin - margin, np.floor), ymin=snapy(ymin - margin, np.floor),
                xmax=snap(xmax + margin, np.ceil), ymax=snapy(ymax + margin, np.ceil), res=r)


# =============================================================================
# Step 2: reference PDAL pipeline
# =============================================================================
def pdal_pipeline(tiles: list[str], sub: Grid, out: Path) -> dict:
    return {"pipeline": [
        *tiles,
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "writers.gdal", "filename": str(out),
         "resolution": sub.res, "binmode": True, "output_type": "mean",
         "origin_x": sub.xmin, "origin_y": sub.ymin,
         "width": sub.width, "height": sub.height,
         "data_type": "float32", "nodata": NODATA,
         "override_srs": sub.crs},
    ]}


# =============================================================================
# Step 3: reading our DTM on a sub-grid, and a resampled reference
# =============================================================================
def read_on(path: Path, target: Grid, resampling=Resampling.nearest) -> np.ndarray:
    """Reads a raster (any projection) resampled on `target`."""
    dst = np.full((target.height, target.width), np.nan, np.float32)
    transform = rasterio.transform.from_origin(target.xmin, target.ymax, target.res, target.res)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=transform, dst_crs=target.crs, resampling=resampling)
    return dst


# =============================================================================
# Step 4: difference statistics
# =============================================================================
def diff_stats(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    d = (ours - ref)[mask & np.isfinite(ours) & np.isfinite(ref)]
    med = np.median(d)
    return {"n": int(d.size),
            "bias_m": float(d.mean()),
            "rmse_m": float(np.sqrt(np.mean(d ** 2))),
            "nmad_m": float(1.4826 * np.median(np.abs(d - med))),
            "p99_abs_m": float(np.percentile(np.abs(d), 99))}


def compare(name: str, ref_path: Path, target: Grid, measured: np.ndarray,
            ours_path: Path, resampling, out_dir: Path) -> None:
    ours = read_on(ours_path, target)
    ref = read_on(ref_path, target, resampling)
    for label, m in [("measured ground", measured), ("interpolated", ~measured)]:
        s = diff_stats(ours, ref, m)
        print(f"  {name:6s} | {label:15s} | n={s['n']:7d}  bias={s['bias_m']:+.3f}  "
              f"RMSE={s['rmse_m']:.3f}  NMAD={s['nmad_m']:.3f}  |d|p99={s['p99_abs_m']:.3f} m")
    diff = np.where(np.isfinite(ours) & np.isfinite(ref), ours - ref, NODATA).astype(np.float32)
    profile = dict(driver="GTiff", width=target.width, height=target.height, count=1,
                   dtype="float32", crs=target.crs, nodata=NODATA, compress="deflate",
                   transform=rasterio.transform.from_origin(target.xmin, target.ymax,
                                                            target.res, target.res))
    with rasterio.open(out_dir / f"diff_dtm_{name}.tif", "w", **profile) as dst:
        dst.write(diff, 1)


# =============================================================================
# Step 5: orchestration
# =============================================================================
def main(cmd: str) -> None:
    cfg = load_config()
    v, rc = cfg["validation"], cfg["rasters"]
    grid = load_grid(ROOT / cfg["site"]["grid_yaml"])
    out_dir = ROOT / v["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rast = ROOT / rc["out_dir"]

    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter")
    sub = subgrid(grid, per.total_bounds, v["margin_m"])

    if cmd == "pdal":
        tiles = sorted(str(p) for p in (ROOT / "data/raw").glob("*.LAZ")
                       if not p.name.lower().endswith(".copc.laz") and tile_intersects(str(p), sub))
        path = out_dir / "dtm_pdal.json"
        path.write_text(json.dumps(pdal_pipeline(tiles, sub, ROOT / v["pdal_ref"]), indent=2))
        print(f"Pipeline written: {path}\nRun: pdal pipeline {path.relative_to(ROOT)}")
        return

    print(f"PDAL test area: {sub.width} x {sub.height} px | full grid: {grid.width} x {grid.height} px")
    refs = [("pdal", ROOT / v["pdal_ref"], sub, Resampling.nearest),
            ("mrnf", ROOT / v["mrnf_ref"], grid, Resampling.bilinear)]
    for name, path, target, resampling in refs:
        if not path.exists():
            print(f"  {name:6s} | reference missing ({path.relative_to(ROOT)}), skipped")
            continue
        measured = read_on(rast / "dtm_filled.tif", target) == 0
        compare(name, path, target, measured, rast / "dtm.tif", resampling, out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "compare")
