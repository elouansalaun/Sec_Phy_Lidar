"""
src/site.py — Définition et dérivation du site à protéger.

Entrée  : data/site/site.gpkg (perimetre, zones, acces), dessiné dans QGIS.
Sorties : site_derive.gpkg, grid.yaml, poids.csv, site.png (chemins dans config.yaml).

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
LAYERS = ("perimetre", "zones", "acces")
VALID_TYPES = {"batiment", "equipement", "peripherie"}
ASSET_TYPES = {"batiment", "equipement"}



# =============================================================================
# Étape 1 — Configuration et chargement
# =============================================================================
def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize(text) -> str:
    """'Équipement ' -> 'equipement' : minuscules, sans accents ni espaces."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower()

def load_site(gpkg: Path) -> dict[str, gpd.GeoDataFrame]:
    layers = {name: gpd.read_file(gpkg, layer=name) for name in LAYERS}
    zones = layers["zones"]
    zones["type"] = zones["type"].map(_normalize)
    zones["cible_intrus"] = zones["cible_intrus"].fillna(False).astype(bool)
    return layers





# =============================================================================
# Étape 2 — Validation
# =============================================================================
class SiteValidationError(ValueError):
    """Le site dessiné viole une hypothèse du modèle."""


def validate_site(layers: dict[str, gpd.GeoDataFrame], crs: str, tol: float) -> None:
    errors: list[str] = []
    per, zones, acces = layers["perimetre"], layers["zones"], layers["acces"]
    expected_epsg = CRS.from_user_input(crs).to_epsg()

    for name, gdf in layers.items():
        if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
            errors.append(f"{name} : SCR {gdf.crs} au lieu de {crs}")
        invalid = ~gdf.is_valid
        if invalid.any():
            errors.append(f"{name} : géométries invalides aux index {gdf.index[invalid].tolist()}")

    if len(per) != 1:
        errors.append(f"perimetre : {len(per)} entités, 1 attendue")

    required = ["nom", "niveau", "poids", "type"]
    if zones[required].isna().any().any():
        errors.append("zones : valeurs manquantes dans " + str(required))
    if (zones["poids"] <= 0).any():
        errors.append("zones : tous les poids doivent être > 0")
    unknown = set(zones["type"]) - VALID_TYPES
    if unknown:
        errors.append(f"zones : types inconnus {unknown}, attendus {VALID_TYPES}")
    if not zones["cible_intrus"].any():
        errors.append("zones : aucune zone n'a cible_intrus = True")

    if len(per) == 1:
        P = per.geometry.iloc[0]
        union = zones.union_all()
        gap = union.symmetric_difference(P).area
        overlap = zones.area.sum() - union.area
        if gap > tol:
            errors.append(f"partition : {gap:.2f} m² d'écart entre zones et périmètre")
        if overlap > tol:
            errors.append(f"partition : {overlap:.2f} m² de chevauchement entre zones")
        if len(acces):
            far = acces.distance(P.boundary) > tol
            if far.any():
                errors.append(f"acces : points hors clôture {acces.loc[far, 'nom'].tolist()}")

    if errors:
        raise SiteValidationError("Site invalide :\n  - " + "\n  - ".join(errors))



# =============================================================================
# Étape 3 — Actifs (cibles de l'intrus, obstacles)
# =============================================================================
def build_assets(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    assets = zones[zones["type"].isin(ASSET_TYPES)].copy()
    assets["surface_m2"] = assets.area
    return assets.reset_index(drop=True)



# =============================================================================
# Étape 4 — Surfaces à surveiller : anneaux d'approche + périphérie
# =============================================================================
def _polygonal(geom, min_area: float = 1.0):
    """Garde seulement les parties surfaciques > min_area (supprime lignes et éclats)."""
    parts = [p for p in shapely.get_parts(geom)
             if p.geom_type in ("Polygon", "MultiPolygon") and p.area >= min_area]
    return shapely.MultiPolygon(
        [q for p in parts for q in shapely.get_parts(p)]
    ) if parts else shapely.MultiPolygon()


def build_coverage_zones(assets: gpd.GeoDataFrame, zones: gpd.GeoDataFrame,
                         perimeter, ring_m: float) -> gpd.GeoDataFrame:
    ground = perimeter.difference(assets.union_all())   # sol surveillable
    order = assets.sort_values(["poids", "niveau", "nom"], ascending=[False, False, True])

    rows, claimed = [], shapely.Polygon()
    for _, a in order.iterrows():
        ring = a.geometry.buffer(ring_m).intersection(ground).difference(claimed)
        ring = _polygonal(ring)
        if ring.is_empty:
            continue
        rows.append({"source": a["nom"], "role": "approche", "niveau": a["niveau"],
                     "poids": a["poids"], "geometry": ring})
        claimed = claimed.union(ring)

    w_periph = zones.loc[zones["type"] == "peripherie", "poids"].max()
    rows.append({"source": "peripherie", "role": "peripherie", "niveau": 1,
                 "poids": w_periph, "geometry": _polygonal(ground.difference(claimed))})
    return gpd.GeoDataFrame(rows, crs=assets.crs)



# =============================================================================
# Étape 5 — Zone d'étude, couronne de l'intrus, grille
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
            f"Emprise {grid.xmin, grid.ymin, grid.xmax, grid.ymax} hors des données "
            f"{tuple(data_extent)} : réduis study_buffer_m ou ajoute des tuiles.")
    couronne = zone.difference(perimeter)
    return zone, couronne, grid



# =============================================================================
# Étape 6 — Tableau des poids
# =============================================================================
def weights_table(cover: gpd.GeoDataFrame) -> pd.DataFrame:
    t = pd.DataFrame(cover.drop(columns="geometry"))
    t["surface_m2"] = cover.area
    t["poids_x_surface"] = t["poids"] * t["surface_m2"]
    t["part_objectif_pct"] = 100 * t["poids_x_surface"] / t["poids_x_surface"].sum()
    return t.sort_values("part_objectif_pct", ascending=False).round(1)



# =============================================================================
# Étape 7 — Figure de contrôle
# =============================================================================
def plot_site(per, assets, cover, zone, acces, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    gpd.GeoSeries([zone], crs=assets.crs).plot(ax=ax1, facecolor="0.95", edgecolor="0.5")
    per.plot(ax=ax1, facecolor="none", edgecolor="k", lw=1.5)
    ax1.set_title("Zone d'étude (buffer) et site")

    cover.plot(ax=ax2, column="poids", cmap="YlOrRd", legend=True,
               legend_kwds={"label": "poids w (surface à surveiller)"}, edgecolor="white", lw=0.3)
    assets.plot(ax=ax2, color="0.3")
    per.boundary.plot(ax=ax2, color="k", lw=1.5)
    if len(acces):
        acces.plot(ax=ax2, color="tab:blue", marker="^", markersize=60, zorder=5)
    ax2.set_title("Surfaces pondérées, actifs (gris) et accès (▲)")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)



# =============================================================================
# Étape 8 — Orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    s, crs = cfg["site"], cfg["lidar"]["crs"]

    layers = load_site(ROOT / s["gpkg"])
    validate_site(layers, crs, s["tolerance_m"])
    per_gdf, zones, acces = layers["perimetre"], layers["zones"], layers["acces"]
    perimeter = per_gdf.geometry.iloc[0]

    assets = build_assets(zones)
    cover = build_coverage_zones(assets, zones, perimeter, s["approach_ring_m"])
    zone, couronne, grid = build_study_area(perimeter, s["study_buffer_m"],
                                            s["resolution_m"], crs,
                                            cfg["lidar"]["data_extent"])

    out = ROOT / s["output_gpkg"]
    out.unlink(missing_ok=True)
    assets.to_file(out, layer="actifs")
    cover.to_file(out, layer="couverture")
    acces.to_file(out, layer="acces")
    gpd.GeoDataFrame({"nom": ["zone_etude", "couronne"]},
                     geometry=[zone, couronne], crs=crs).to_file(out, layer="emprises")

    with open(ROOT / s["grid_yaml"], "w", encoding="utf-8") as f:
        yaml.safe_dump(grid.to_dict(), f, sort_keys=False)

    table = weights_table(cover)
    table.to_csv(ROOT / s["weights_csv"], index=False)
    plot_site(per_gdf, assets, cover, zone, acces, ROOT / s["figure"])

    print(table.to_string(index=False))
    print(f"\nGrille : {grid.width} x {grid.height} px à {grid.res} m -> {grid.to_dict()}")
    print(f"Sorties écrites dans {out.parent}")


if __name__ == "__main__":
    main()