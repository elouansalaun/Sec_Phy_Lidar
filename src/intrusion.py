"""
src/intrusion.py — Domaine de l'intrus : cellules, graphe, matrice de détection (formulation §7.2).

Le domaine = cellules sol intérieures (celles de cellules.npz, même ordre) + cellules sol
d'une bande extérieure de largeur b. La clôture est portée par les arêtes : une arête
« franchit » si ses deux cellules sont de part et d'autre du périmètre, et elle passe
« par un portail » si son segment coupe l'ouverture de ce portail.

A_intrus est calculée par la même fonction que A (coverage.build_matrix) : sur les colonnes
intérieures, elle doit reproduire A exactement (contrôle de non-régression).

Sorties (intrus.out_dir) : domaine.npz, graphe.npz, A_intrus.npz, configs_intrus.csv,
                           portails.csv, domaine.png
Usage : python -m src.intrusion
"""
from __future__ import annotations

import re
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import sparse

from src.coverage import build_matrix
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces

SQRT2 = np.sqrt(2.0)
# 4 demi-voisinages : chaque arête non orientée n'est générée qu'une fois
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


# =============================================================================
# Étape 1 — Cellules du domaine
# =============================================================================
def centers(grid, rows, cols) -> np.ndarray:
    return shapely.points(grid.xmin + (cols + 0.5) * grid.res,
                          grid.ymax - (rows + 0.5) * grid.res)


def burn(geoms, grid) -> np.ndarray:
    """Masque booléen, règle du centre de pixel (comme coverage.weighted_cells)."""
    return rasterize(((g, 1) for g in geoms), out_shape=(grid.height, grid.width),
                     transform=from_origin(grid.xmin, grid.ymax, grid.res, grid.res),
                     fill=0, dtype="uint8").astype(bool)


def domain_cells(s, per, assets, cells_s3, cfg):
    g, ic = s.grid, cfg["intrus"]
    b, d_t = float(ic["bande_ext_m"]), float(ic["cible_contact_m"])
    fence = per.boundary

    inside = burn([per], g)
    asset = burn(assets.geometry, g)
    sol = s.z_opaque < s.z_tgt                        # un intrus peut s'y tenir (NaN -> False)

    # Intérieur : exactement les cellules de la semaine 3, dans le même ordre
    ri, ci = cells_s3["rows"], cells_s3["cols"]
    assert inside[ri, ci].all() and sol[ri, ci].all() and not asset[ri, ci].any(), \
        "cellules.npz incohérent avec le périmètre, le sol ou les actifs"
    oubli = inside & sol & ~asset
    oubli[ri, ci] = False
    n_oubli = int(oubli.sum())                        # sol intérieur hors zones : traité en obstacle

    # Extérieur : sol hors périmètre, à moins de b de la clôture
    band = burn([per.buffer(b + g.res)], g) & ~inside & sol
    re_, ce_ = np.nonzero(band)
    d_ext = shapely.distance(centers(g, re_, ce_), fence)
    keep = d_ext <= b
    re_, ce_, d_ext = re_[keep], ce_[keep], d_ext[keep]

    rows = np.concatenate([ri, re_])
    cols = np.concatenate([ci, ce_])
    interieur = np.r_[np.ones(ri.size, bool), np.zeros(re_.size, bool)]
    pts = centers(g, rows, cols)
    dist = np.r_[-shapely.distance(pts[:ri.size], fence), d_ext]   # < 0 à l'intérieur

    target = assets.loc[assets["cible_intrus"].astype(bool)]
    assert len(target) == 1, f"{len(target)} actifs cible_intrus, 1 attendu"
    tgt = target.geometry.iloc[0]
    cible = interieur & (shapely.distance(pts, tgt) <= d_t)        # jamais une cellule extérieure
    source = ~interieur & (dist >= b - 1.5 * g.res)
    return dict(rows=rows, cols=cols, interieur=interieur, dist_cloture=dist.astype(np.float32),
                source=source, cible=cible, cible_nom=target["nom"].iloc[0],
                n_oubli=n_oubli, n_ext_cible_proche=int(((~interieur) &
                                  (shapely.distance(pts, tgt) <= d_t)).sum()))


# =============================================================================
# Étape 2 — Graphe en 8-connexité
# =============================================================================
def build_edges(grid, rows, cols, interieur):
    idx = np.full((grid.height, grid.width), -1, np.int64)
    idx[rows, cols] = np.arange(rows.size)
    H, W = idx.shape
    U, V, L = [], [], []
    for dr, dc in OFFSETS:
        r0, r1 = 0, H - dr
        c0, c1 = max(0, -dc), W - max(0, dc)
        a = idx[r0:r1, c0:c1]
        bb = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ok = (a >= 0) & (bb >= 0)
        if dr and dc:
            # diagonale interdite si les deux voisins orthogonaux sont des obstacles
            o1 = idx[r0 + dr:r1 + dr, c0:c1]
            o2 = idx[r0:r1, c0 + dc:c1 + dc]
            ok &= (o1 >= 0) | (o2 >= 0)
        U.append(a[ok]); V.append(bb[ok])
        L.append(np.full(ok.sum(), grid.res * (SQRT2 if dr and dc else 1.0)))
    u, v, l = np.concatenate(U), np.concatenate(V), np.concatenate(L)
    return u, v, l, interieur[u] != interieur[v]


# =============================================================================
# Étape 3 — Portails : ouverture = segment de clôture centré sur le point d'accès
# =============================================================================
def gate_width(type_str: str) -> float:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\b", str(type_str))
    if m is None:
        raise ValueError(f"largeur absente du type de portail « {type_str} » (ex. « Portail 3m »)")
    return float(m.group(1).replace(",", "."))


def gate_edges(grid, rows, cols, u, v, franchit, per, acces):
    fence = per.boundary
    portail = np.full(u.size, -1, np.int16)
    ix = np.flatnonzero(franchit)
    p = centers(grid, rows, cols)
    xy = shapely.get_coordinates(p)
    seg = shapely.linestrings(np.stack([xy[u[ix]], xy[v[ix]]], axis=1))
    table = []
    for k, a in enumerate(acces.itertuples()):
        assert a.geometry.distance(fence) < 0.5, f"portail {a.nom} hors de la clôture"
        w = gate_width(a.type)
        opening = fence.intersection(a.geometry.buffer(w / 2))
        hit = shapely.intersects(seg, opening)
        assert (portail[ix[hit]] == -1).all(), f"portail {a.nom} chevauche un autre portail"
        portail[ix[hit]] = k
        if not hit.any():
            raise ValueError(f"portail {a.nom} : aucune arête franchissable (obstacle devant ?)")
        table.append(dict(portail=k, nom=a.nom, type=a.type, largeur_m=w,
                          ouverture_m=round(opening.length, 2), n_aretes=int(hit.sum())))
    return portail, pd.DataFrame(table)


# =============================================================================
# Étape 4 — Matrice de détection sur le domaine, rattachée aux configurations S3
# =============================================================================
KEY = ["cand", "h", "capteur", "azimut", "inclinaison"]


def config_key(df: pd.DataFrame) -> pd.Series:
    k = df[KEY].copy()
    k["azimut"] = k["azimut"].fillna(-1.0)
    k["inclinaison"] = k["inclinaison"].round(6)
    return k.astype(str).agg("|".join, axis=1)


def match_s3(conf: pd.DataFrame, conf_s3: pd.DataFrame) -> np.ndarray:
    pos = pd.Series(np.arange(len(conf_s3)), index=config_key(conf_s3))
    assert pos.index.is_unique
    return config_key(conf).map(pos).fillna(-1).to_numpy(np.int64)


def check_non_regression(A_int, A_s3, config_s3, n_int) -> float:
    """A_intrus restreinte aux cellules intérieures doit reproduire A ligne à ligne."""
    have = config_s3 >= 0
    assert np.isin(np.arange(A_s3.shape[0]), config_s3[have]).all(), \
        "des configurations S3 manquent dans A_intrus"
    sub = A_int[np.flatnonzero(have)][:, :n_int]
    ref = A_s3[config_s3[have]]
    diff = abs(sub - ref)
    return float(diff.max()) if diff.nnz else 0.0


# =============================================================================
# Étape 5 — Figure de contrôle
# =============================================================================
INK, INK_2 = "#0b0b0b", "#52514e"
C_SRC, C_TGT, C_FENCE, C_GATE = "#1baf7a", "#eb6834", "#0b0b0b", "#4a3aa7"
C_INT, C_BAND, C_OBST = "#dfe9f7", "#f1f0ec", "#b9b7b0"


def plot_domain(grid, dom, u, v, portail, per, assets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    r, c = dom["rows"], dom["cols"]
    img = np.full((grid.height, grid.width), np.nan)
    img[r, c] = dom["interieur"].astype(float)          # 0 = bande extérieure, 1 = intérieur
    r0, r1, c0, c1 = r.min() - 3, r.max() + 4, c.min() - 3, c.max() + 4
    ext = (grid.xmin + c0 * grid.res, grid.xmin + c1 * grid.res,
           grid.ymax - r1 * grid.res, grid.ymax - r0 * grid.res)
    X = grid.xmin + (c + 0.5) * grid.res
    Y = grid.ymax - (r + 0.5) * grid.res

    fig, ax = plt.subplots(figsize=(8, 8 * (r1 - r0) / (c1 - c0) + 0.9), dpi=150)
    ax.imshow(img[r0:r1, c0:c1], extent=ext, cmap=ListedColormap([C_BAND, C_INT]),
              vmin=0, vmax=1, interpolation="nearest")
    gpd.GeoSeries(list(assets.geometry)).plot(ax=ax, color=C_OBST, edgecolor=INK_2, lw=0.5)
    gpd.GeoSeries([per.boundary]).plot(ax=ax, color=C_FENCE, lw=1.2)
    ax.scatter(X[dom["source"]], Y[dom["source"]], s=0.6, color=C_SRC, lw=0)
    ax.scatter(X[dom["cible"]], Y[dom["cible"]], s=2, color=C_TGT, lw=0)
    gt = portail >= 0
    ax.scatter((X[u[gt]] + X[v[gt]]) / 2, (Y[u[gt]] + Y[v[gt]]) / 2, s=14, color=C_GATE, lw=0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(color=C_INT, label="Intérieur"),
                       Patch(color=C_BAND, label=f"Bande extérieure ({dom['b']:g} m)"),
                       Patch(color=C_OBST, label="Actifs (obstacles)"),
                       Line2D([], [], color=C_FENCE, lw=1.2, label="Clôture"),
                       Line2D([], [], marker="o", ls="", color=C_SRC, label="Départ de l'intrus"),
                       Line2D([], [], marker="o", ls="", color=C_TGT,
                              label=f"Cible ({dom['cible_nom']})"),
                       Line2D([], [], marker="o", ls="", color=C_GATE,
                              label="Franchissements par portail")],
              frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=4, labelcolor=INK_2)
    ax.set_title("Domaine de l'intrus", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Étape 6 — Orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    ic = cfg["intrus"]
    out = ROOT / ic["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cov = ROOT / cfg["coverage"]["out_dir"]

    s = prepare_surfaces(cfg)
    g = s.grid
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    derive = ROOT / cfg["site"]["output_gpkg"]
    assets = gpd.read_file(derive, layer="actifs")
    acces = gpd.read_file(derive, layer="acces")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    cells_s3 = np.load(cov / "cellules.npz")
    A_s3 = sparse.load_npz(cov / "A.npz").tocsr()
    conf_s3 = pd.read_csv(cov / "configs.csv")

    # --- Cellules
    dom = domain_cells(s, per, assets, cells_s3, cfg)
    dom["b"] = float(ic["bande_ext_m"])
    n_int = int(dom["interieur"].sum())
    print(f"Domaine : {dom['rows'].size:,} cellules = {n_int:,} intérieures "
          f"+ {dom['rows'].size - n_int:,} extérieures (bande de {dom['b']:g} m)")
    print(f"  sources : {dom['source'].sum():,}   cibles ({dom['cible_nom']}) : "
          f"{dom['cible'].sum():,}")
    print(f"  cellules extérieures à ≤ {ic['cible_contact_m']} m de la cible, exclues des "
          f"cibles : {dom['n_ext_cible_proche']}")
    print(f"  sol intérieur hors zones pondérées (obstacles) : {dom['n_oubli']}")

    # --- Graphe
    u, v, l, franchit = build_edges(g, dom["rows"], dom["cols"], dom["interieur"])
    portail, gates = gate_edges(g, dom["rows"], dom["cols"], u, v, franchit, per, acces)
    print(f"\nGraphe : {u.size:,} arêtes, dont {franchit.sum():,} franchissent la clôture "
          f"et {(portail >= 0).sum()} par un portail")
    print(gates.to_string(index=False))

    # --- Matrice
    t0 = time.time()
    w_dom = np.r_[cells_s3["w"], np.zeros(dom["rows"].size - n_int)]
    A, conf, n_empty, lost = build_matrix(s, cand, dom["rows"], dom["cols"], w_dom, cfg)
    conf["config_s3"] = match_s3(conf, conf_s3)
    err = check_non_regression(A, A_s3, conf["config_s3"].to_numpy(), n_int)
    print(f"\nA_intrus : {A.shape[0]:,} × {A.shape[1]:,}, {A.nnz:,} non nuls "
          f"({time.time() - t0:.0f} s) ; {n_empty} configurations sans aucune détection")
    new = conf["config_s3"] < 0
    print(f"  configurations S3 retrouvées : {(~new).sum():,} / {len(conf_s3):,} ; "
          f"nouvelles (ne voient que l'extérieur) : {new.sum():,}")
    print(f"  non-régression A_intrus[:, intérieur] vs A : écart max = {err:.2e}")
    assert err < 1e-6, "A_intrus ne reproduit pas A sur l'intérieur"
    print("  nouvelles configurations par capteur :",
          conf.loc[new, "capteur"].value_counts().to_dict())

    p_max = np.asarray(A.max(axis=0).todense()).ravel()
    ext = ~dom["interieur"]
    near = ext & (dom["dist_cloture"] <= 5)
    print(f"\nP max (meilleure configuration isolée) dans la bande : médiane "
          f"{np.median(p_max[ext]):.2f}, part < 0,5 : {(p_max[ext] < 0.5).mean():.1%} ; "
          f"à ≤ 5 m de la clôture : part < 0,5 : {(p_max[near] < 0.5).mean():.1%}")

    # --- Sauvegarde
    np.savez(out / "domaine.npz", **{k: dom[k] for k in
             ("rows", "cols", "interieur", "dist_cloture", "source", "cible")}, w=w_dom)
    np.savez(out / "graphe.npz", u=u, v=v, longueur=l, franchit=franchit, portail=portail)
    sparse.save_npz(out / "A_intrus.npz", A)
    conf.to_csv(out / "configs_intrus.csv", index=False)
    gates.to_csv(out / "portails.csv", index=False)
    plot_domain(g, dom, u, v, portail, per, assets, out / "domaine.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
