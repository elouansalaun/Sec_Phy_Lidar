"""
src/site.py: definition and derivation of the site to protect.

Input   : data/site/site.gpkg (perimeter, zones, access), drawn in QGIS.
Outputs : site_derived.gpkg, grid.yaml, weights.csv, site.png (paths in config.yaml).

Usage   : python -m src.site
"""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("perimeter", "zones", "access")
VALID_TYPES = {"building", "equipment", "periphery"}
ASSET_TYPES = {"building", "equipment"}

# Sites drawn with an early version of the project use French layer, field and type names.
# They are read transparently; new sites should use the English schema above.
LEGACY_LAYERS = {"perimetre": "perimeter", "acces": "access"}
LEGACY_FIELDS = {"nom": "name", "niveau": "level", "poids": "weight",
                 "cible_intrus": "intruder_target", "type_cloture": "fence_type"}
LEGACY_TYPES = {"batiment": "building", "equipement": "equipment", "peripherie": "periphery"}



# =============================================================================
# Step 1: configuration and loading
# =============================================================================
def load_config(path: Path = ROOT / "config.yaml") -> dict:
    """config.yaml; with sensors.cameras_only, the radars are removed for every stage of the chain."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["sensors"].get("cameras_only", False):
        cfg["sensors"]["radars"] = []
    return cfg


def _normalize(text) -> str:
    """'Équipement ' -> 'equipement': lower case, no accents or spaces."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower()


def read_layer(gpkg: Path, name: str) -> gpd.GeoDataFrame:
    """One layer of a site GeoPackage, legacy (French) layer and field names translated."""
    import pyogrio

    available = {n for n, _ in pyogrio.list_layers(gpkg)}
    legacy = {v: k for k, v in LEGACY_LAYERS.items()}.get(name)
    layer = name if name in available or legacy not in available else legacy
    gdf = gpd.read_file(gpkg, layer=layer)
    return gdf.rename(columns={k: v for k, v in LEGACY_FIELDS.items() if k in gdf.columns})


def load_site(gpkg: Path) -> dict[str, gpd.GeoDataFrame]:
    layers = {name: read_layer(gpkg, name) for name in LAYERS}
    zones = layers["zones"]
    zones["type"] = zones["type"].map(_normalize).replace(LEGACY_TYPES)
    zones["intruder_target"] = zones["intruder_target"].fillna(False).astype(bool)
    return layers





# =============================================================================
# Step 2: validation
# =============================================================================
class SiteValidationError(ValueError):
    """The drawn site violates an assumption of the model."""


def validate_site(layers: dict[str, gpd.GeoDataFrame], crs: str, tol: float) -> None:
    errors: list[str] = []
    per, zones, access = layers["perimeter"], layers["zones"], layers["access"]
    expected_epsg = CRS.from_user_input(crs).to_epsg()

    for name, gdf in layers.items():
        if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
            errors.append(f"{name}: CRS {gdf.crs} instead of {crs}")
        invalid = ~gdf.is_valid
        if invalid.any():
            errors.append(f"{name}: invalid geometries at indices {gdf.index[invalid].tolist()}")

    if len(per) != 1:
        errors.append(f"perimeter: {len(per)} features, 1 expected")

    required = ["name", "level", "weight", "type"]
    if zones[required].isna().any().any():
        errors.append("zones: missing values in " + str(required))
    if (zones["weight"] <= 0).any():
        errors.append("zones: every weight must be > 0")
    unknown = set(zones["type"]) - VALID_TYPES
    if unknown:
        errors.append(f"zones: unknown types {unknown}, expected {VALID_TYPES}")
    if not zones["intruder_target"].any():
        errors.append("zones: no zone has intruder_target = True")

    if len(per) == 1:
        P = per.geometry.iloc[0]
        union = zones.union_all()
        gap = union.symmetric_difference(P).area
        overlap = zones.area.sum() - union.area
        if gap > tol:
            errors.append(f"partition: {gap:.2f} m² of gap between zones and perimeter")
        if overlap > tol:
            errors.append(f"partition: {overlap:.2f} m² of overlap between zones")
        if len(access):
            far = access.distance(P.boundary) > tol
            if far.any():
                errors.append(f"access: points off the fence {access.loc[far, 'name'].tolist()}")

    if errors:
        raise SiteValidationError("Invalid site:\n  - " + "\n  - ".join(errors))



# =============================================================================
# Step 3: assets (intruder targets, obstacles)
# =============================================================================
def build_assets(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    assets = zones[zones["type"].isin(ASSET_TYPES)].copy()
    assets["area_m2"] = assets.area
    return assets.reset_index(drop=True)



# =============================================================================
# Step 4: surfaces to watch: approach rings + periphery
# =============================================================================
def _polygonal(geom, min_area: float = 1.0):
    """Keeps only the areal parts > min_area (removes lines and slivers)."""
    parts = [p for p in shapely.get_parts(geom)
             if p.geom_type in ("Polygon", "MultiPolygon") and p.area >= min_area]
    return shapely.MultiPolygon(
        [q for p in parts for q in shapely.get_parts(p)]
    ) if parts else shapely.MultiPolygon()


def build_coverage_zones(assets: gpd.GeoDataFrame, zones: gpd.GeoDataFrame,
                         perimeter, ring_m: float) -> gpd.GeoDataFrame:
    ground = perimeter.difference(assets.union_all())   # ground that can be watched
    order = assets.sort_values(["weight", "level", "name"], ascending=[False, False, True])

    rows, claimed = [], shapely.Polygon()
    for _, a in order.iterrows():
        ring = a.geometry.buffer(ring_m).intersection(ground).difference(claimed)
        ring = _polygonal(ring)
        if ring.is_empty:
            continue
        rows.append({"source": a["name"], "role": "approach", "level": a["level"],
                     "weight": a["weight"], "geometry": ring})
        claimed = claimed.union(ring)

    w_periph = zones.loc[zones["type"] == "periphery", "weight"].max()
    rows.append({"source": "periphery", "role": "periphery", "level": 1,
                 "weight": w_periph, "geometry": _polygonal(ground.difference(claimed))})
    return gpd.GeoDataFrame(rows, crs=assets.crs)



# =============================================================================
# Step 5: study area, intruder ring, grid
# =============================================================================
@dataclass
class Grid:
    crs: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    def to_dict(self) -> dict:
        return {**asdict(self), "width": self.width, "height": self.height}


def buffer_auto(perimeter, data_extent, res: float, b_min: float, b_max: float) -> float:
    """Largest buffer <= b_max that fits in the data (site close to the edge of a tile).

    Visibility rays only connect sensors of the site to cells of the intruder domain
    (site + outer_band_m): a buffer of a few tens of metres is enough for the computation.
    """
    x0, y0, x1, y1 = perimeter.bounds
    dx0, dy0, dx1, dy1 = data_extent
    margin = np.floor(min(x0 - dx0, y0 - dy0, dx1 - x1, dy1 - y1) / res) * res - res
    if margin < b_min:
        raise SiteValidationError(f"the site is {margin:.0f} m from the edge of the LiDAR data "
                                  f"(minimum {b_min:g} m): add neighbouring tiles.")
    return float(min(b_max, margin))


def build_study_area(perimeter, buffer_m: float, res: float, crs: str,
                     data_extent: list[float]):
    zone = perimeter.buffer(buffer_m)
    xmin, ymin, xmax, ymax = zone.bounds
    grid = Grid(crs=crs,
                xmin=float(np.floor(xmin / res) * res), ymin=float(np.floor(ymin / res) * res),
                xmax=float(np.ceil(xmax / res) * res), ymax=float(np.ceil(ymax / res) * res),
                res=res)
    dx0, dy0, dx1, dy1 = data_extent
    if grid.xmin < dx0 or grid.ymin < dy0 or grid.xmax > dx1 or grid.ymax > dy1:
        raise SiteValidationError(
            f"Extent {grid.xmin, grid.ymin, grid.xmax, grid.ymax} outside the data "
            f"{tuple(data_extent)}: reduce study_buffer_m or add tiles.")
    ring = zone.difference(perimeter)
    return zone, ring, grid



# =============================================================================
# Step 6: weight table
# =============================================================================
def weights_table(cover: gpd.GeoDataFrame) -> pd.DataFrame:
    t = pd.DataFrame(cover.drop(columns="geometry"))
    t["area_m2"] = cover.area
    t["weight_x_area"] = t["weight"] * t["area_m2"]
    t["objective_share_pct"] = 100 * t["weight_x_area"] / t["weight_x_area"].sum()
    return t.sort_values("objective_share_pct", ascending=False).round(1)



# =============================================================================
# Step 7: check figure
# =============================================================================
def plot_site(per, assets, cover, zone, access, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    gpd.GeoSeries([zone], crs=assets.crs).plot(ax=ax1, facecolor="0.95", edgecolor="0.5")
    per.plot(ax=ax1, facecolor="none", edgecolor="k", lw=1.5)
    ax1.set_title("Study area (buffer) and site")

    cover.plot(ax=ax2, column="weight", cmap="YlOrRd", legend=True,
               legend_kwds={"label": "weight w (surface to watch)"}, edgecolor="white", lw=0.3)
    assets.plot(ax=ax2, color="0.3")
    per.boundary.plot(ax=ax2, color="k", lw=1.5)
    if len(access):
        access.plot(ax=ax2, color="tab:blue", marker="^", markersize=60, zorder=5)
    ax2.set_title("Weighted surfaces, assets (grey) and access points (▲)")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)



# =============================================================================
# Step 8: orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    s, crs = cfg["site"], cfg["lidar"]["crs"]

    layers = load_site(ROOT / s["gpkg"])
    validate_site(layers, crs, s["tolerance_m"])
    per_gdf, zones, access = layers["perimeter"], layers["zones"], layers["access"]
    perimeter = per_gdf.geometry.iloc[0]

    assets = build_assets(zones)
    cover = build_coverage_zones(assets, zones, perimeter, s["approach_ring_m"])
    buffer_m = s["study_buffer_m"]
    if buffer_m == "auto":
        buffer_m = buffer_auto(perimeter, cfg["lidar"]["data_extent"], s["resolution_m"],
                               cfg["intruder"]["outer_band_m"] + 20, s.get("study_buffer_max_m", 400))
        print(f"Study area buffer (auto): {buffer_m:g} m")
    zone, ring, grid = build_study_area(perimeter, buffer_m,
                                        s["resolution_m"], crs,
                                        cfg["lidar"]["data_extent"])

    out = ROOT / s["output_gpkg"]
    out.unlink(missing_ok=True)
    assets.to_file(out, layer="assets")
    cover.to_file(out, layer="coverage")
    access.to_file(out, layer="access")
    gpd.GeoDataFrame({"name": ["study_area", "ring"]},
                     geometry=[zone, ring], crs=crs).to_file(out, layer="extents")

    with open(ROOT / s["grid_yaml"], "w", encoding="utf-8") as f:
        yaml.safe_dump(grid.to_dict(), f, sort_keys=False)

    table = weights_table(cover)
    table.to_csv(ROOT / s["weights_csv"], index=False)
    plot_site(per_gdf, assets, cover, zone, access, ROOT / s["figure"])

    print(table.to_string(index=False))
    print(f"\nGrid: {grid.width} x {grid.height} px at {grid.res} m -> {grid.to_dict()}")
    print(f"Outputs written to {out.parent}")


if __name__ == "__main__":
    main()
