"""
src/candidates.py: candidate sensor positions (ground masts, rooftop supports).

Inputs : site.gpkg, site_derived.gpkg (assets), DTM / DSM / height rasters.
Output : candidates.gpkg, one row per position, with the support elevation and possible heights.

Usage  : python -m src.candidates
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
import shapely

from src.site import ROOT, load_config


# =============================================================================
# Step 1: sample a raster at points
# =============================================================================
def sample(path, xy: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(xy)], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals


# =============================================================================
# Step 2: masts: along the fence + interior grid
# =============================================================================
def mast_points(perimeter, buildings, c: dict) -> list[tuple[shapely.Point, str]]:
    inner = perimeter.buffer(-c["inset_m"])
    ring = inner.exterior
    n = max(int(ring.length // c["spacing_perimeter_m"]), 1)
    pts = [(ring.interpolate(i * ring.length / n), "fence_mast") for i in range(n)]

    s = c["spacing_interior_m"]
    x0, y0, x1, y1 = inner.bounds
    xs = np.arange(np.ceil(x0 / s) * s, x1, s)
    ys = np.arange(np.ceil(y0 / s) * s, y1, s)
    interior = inner.buffer(-s / 2)                   # avoids duplicates with the fence
    pts += [(shapely.Point(x, y), "interior_mast") for x in xs for y in ys
            if interior.contains(shapely.Point(x, y))]

    forbidden = buildings.buffer(c["min_dist_building_m"])
    return [(p, t) for p, t in pts if not forbidden.contains(p)]

# =============================================================================
# Step 3: roofs: building corners, set back from the edge
# =============================================================================
def roof_points(assets: gpd.GeoDataFrame, c: dict) -> list[tuple[shapely.Point, str, str]]:
    out = []
    for _, a in assets[assets["type"] == "building"].iterrows():
        core = a.geometry.buffer(-c["inset_m"]).simplify(c["simplify_m"])
        for poly in shapely.get_parts(core):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            for x, y in list(poly.exterior.coords)[:-1]:
                out.append((shapely.Point(x, y), "roof", a["name"]))
    return out

# =============================================================================
# Step 4: assembly: elevations, possible heights, costs
# =============================================================================
def build_candidates(perimeter, assets, rasters_dir, cfg_c: dict, crs) -> gpd.GeoDataFrame:
    buildings = assets[assets["type"] == "building"].union_all()
    rows = []
    for p, t in mast_points(perimeter, buildings, cfg_c["mast"]):
        rows.append({"support": t, "building": None, "geometry": p})
    for p, t, b in roof_points(assets, cfg_c["roof"]):
        rows.append({"support": t, "building": b, "geometry": p})
    gdf = gpd.GeoDataFrame(rows, crs=crs)

    xy = np.c_[gdf.geometry.x, gdf.geometry.y]
    is_roof = (gdf["support"] == "roof").to_numpy()
    on_roof = sample(rasters_dir / "building_mask.tif", xy) == 1
    keep = ~is_roof | on_roof                     # corner drawn outside the LiDAR roof -> rejected
    gdf, xy, is_roof = gdf[keep].reset_index(drop=True), xy[keep], is_roof[keep]

    gdf["z_ground"] = sample(rasters_dir / "dtm.tif", xy)
    gdf["z_dsm"] = sample(rasters_dir / "dsm.tif", xy)
    gdf["h_vegetation"] = np.where(is_roof, np.nan, sample(rasters_dir / "other_h.tif", xy))
    gdf["z_support"] = np.where(is_roof, gdf["z_dsm"], gdf["z_ground"])

    mast, roof = cfg_c["mast"], cfg_c["roof"]
    gdf["heights_m"] = np.where(is_roof, ",".join(map(str, roof["heights_m"])),
                                ",".join(map(str, mast["heights_m"])))
    gdf["cost_base"] = np.where(is_roof, roof["cost_base"], mast["cost_base"])
    gdf["cost_per_m"] = np.where(is_roof, roof["cost_per_m"], mast["cost_per_m"])

    gdf = gdf.dropna(subset=["z_support"]).reset_index(drop=True)
    gdf.insert(0, "id", [f"C{i:03d}" for i in range(len(gdf))])
    return gdf

# =============================================================================
# Step 5: orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    s, c = cfg["site"], cfg["candidates"]
    perimeter = gpd.read_file(ROOT / s["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / s["output_gpkg"], layer="assets")
    cand = build_candidates(perimeter, assets, ROOT / cfg["rasters"]["out_dir"], c,
                            cfg["lidar"]["crs"])

    out = ROOT / c["output"]
    out.unlink(missing_ok=True)
    cand.to_file(out, layer="candidates")

    n_conf = sum(len(h.split(",")) for h in cand["heights_m"])
    print(cand.groupby("support").agg(n=("id", "size"), z_support_mean=("z_support", "mean"),
                                      veg_mean=("h_vegetation", "mean")).round(1))
    print(f"\n{len(cand)} positions, {n_conf} (position, height) pairs -> {out}")
    in_trees = (cand["support"] != "roof") & (cand["h_vegetation"] > 3)
    print(f"Masts in vegetation (> 3 m): {int(in_trees.sum())}")


if __name__ == "__main__":
    main()
