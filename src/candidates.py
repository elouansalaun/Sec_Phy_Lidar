"""
src/candidates.py — Positions candidates des capteurs (mâts au sol, supports en toiture).

Entrées : site.gpkg, site_derive.gpkg (actifs), rasters MNT / MNS / hauteurs.
Sortie  : candidats.gpkg — une ligne par position, avec altitude du support et hauteurs possibles.

Usage   : python -m src.candidates
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
import shapely

from src.site import ROOT, load_config


# =============================================================================
# Étape 1 — Échantillonner un raster aux points
# =============================================================================
def sample(path, xy: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(xy)], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals


# =============================================================================
# Étape 2 — Mâts : le long de la clôture + grille intérieure
# =============================================================================
def mast_points(perimeter, buildings, c: dict) -> list[tuple[shapely.Point, str]]:
    inner = perimeter.buffer(-c["inset_m"])
    ring = inner.exterior
    n = max(int(ring.length // c["spacing_perimeter_m"]), 1)
    pts = [(ring.interpolate(i * ring.length / n), "mat_cloture") for i in range(n)]

    s = c["spacing_interior_m"]
    x0, y0, x1, y1 = inner.bounds
    xs = np.arange(np.ceil(x0 / s) * s, x1, s)
    ys = np.arange(np.ceil(y0 / s) * s, y1, s)
    interior = inner.buffer(-s / 2)                   # évite les doublons avec la clôture
    pts += [(shapely.Point(x, y), "mat_interieur") for x in xs for y in ys
            if interior.contains(shapely.Point(x, y))]

    forbidden = buildings.buffer(c["min_dist_building_m"])
    return [(p, t) for p, t in pts if not forbidden.contains(p)]

# =============================================================================
# Étape 3 — Toits : coins des bâtiments, en retrait du bord
# =============================================================================
def roof_points(assets: gpd.GeoDataFrame, c: dict) -> list[tuple[shapely.Point, str, str]]:
    out = []
    for _, a in assets[assets["type"] == "batiment"].iterrows():
        core = a.geometry.buffer(-c["inset_m"]).simplify(c["simplify_m"])
        for poly in shapely.get_parts(core):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            for x, y in list(poly.exterior.coords)[:-1]:
                out.append((shapely.Point(x, y), "toit", a["nom"]))
    return out

# =============================================================================
# Étape 4 — Assemblage : altitudes, hauteurs possibles, coûts
# =============================================================================
def build_candidates(perimeter, assets, rasters_dir, cfg_c: dict, crs) -> gpd.GeoDataFrame:
    buildings = assets[assets["type"] == "batiment"].union_all()
    rows = []
    for p, t in mast_points(perimeter, buildings, cfg_c["mat"]):
        rows.append({"support": t, "batiment": None, "geometry": p})
    for p, t, b in roof_points(assets, cfg_c["toit"]):
        rows.append({"support": t, "batiment": b, "geometry": p})
    gdf = gpd.GeoDataFrame(rows, crs=crs)

    xy = np.c_[gdf.geometry.x, gdf.geometry.y]
    is_roof = (gdf["support"] == "toit").to_numpy()
    on_roof = sample(rasters_dir / "bati_mask.tif", xy) == 1
    keep = ~is_roof | on_roof                     # coin dessiné hors du toit LiDAR -> rejeté
    gdf, xy, is_roof = gdf[keep].reset_index(drop=True), xy[keep], is_roof[keep]

    gdf["z_sol"] = sample(rasters_dir / "mnt.tif", xy)
    gdf["z_mns"] = sample(rasters_dir / "mns.tif", xy)
    gdf["h_vegetation"] = np.where(is_roof, np.nan, sample(rasters_dir / "autre_h.tif", xy))
    gdf["z_support"] = np.where(is_roof, gdf["z_mns"], gdf["z_sol"])

    mat, toit = cfg_c["mat"], cfg_c["toit"]
    gdf["hauteurs_m"] = np.where(is_roof, ",".join(map(str, toit["heights_m"])),
                                 ",".join(map(str, mat["heights_m"])))
    gdf["cout_base"] = np.where(is_roof, toit["cost_base"], mat["cost_base"])
    gdf["cout_par_m"] = np.where(is_roof, toit["cost_per_m"], mat["cost_per_m"])

    gdf = gdf.dropna(subset=["z_support"]).reset_index(drop=True)
    gdf.insert(0, "id", [f"C{i:03d}" for i in range(len(gdf))])
    return gdf

# =============================================================================
# Étape 5 — Orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    s, c = cfg["site"], cfg["candidates"]
    perimeter = gpd.read_file(ROOT / s["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / s["output_gpkg"], layer="actifs")
    cand = build_candidates(perimeter, assets, ROOT / cfg["rasters"]["out_dir"], c,
                            cfg["lidar"]["crs"])

    out = ROOT / c["output"]
    out.unlink(missing_ok=True)
    cand.to_file(out, layer="candidats")

    n_conf = sum(len(h.split(",")) for h in cand["hauteurs_m"])
    print(cand.groupby("support").agg(n=("id", "size"), z_support_moy=("z_support", "mean"),
                                      veg_moy=("h_vegetation", "mean")).round(1))
    print(f"\n{len(cand)} positions, {n_conf} couples (position, hauteur) -> {out}")
    in_trees = (cand["support"] != "toit") & (cand["h_vegetation"] > 3)
    print(f"Mâts dans la végétation (> 3 m) : {int(in_trees.sum())}")


if __name__ == "__main__":
    main()