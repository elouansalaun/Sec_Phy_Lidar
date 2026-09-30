"""
src/intruder_domain.py: domain of the intruder: cells, graph, detection matrix (formulation §7.2).

The domain = interior ground cells (those of cells.npz, same order) + ground cells of an
outer band of width b. The fence is carried by the edges: an edge "crosses" if its two cells
are on either side of the perimeter, and it goes "through a gate" if its segment intersects
the opening of that gate.

A_intruder is computed by the same function as A (coverage.build_matrix): on the interior
columns, it must reproduce A exactly (non-regression check).

Outputs (intruder.out_dir): domain.npz, graph.npz, A_intruder.npz, intruder_configs.csv,
                            gates.csv, domain.png
Usage : python -m src.intruder_domain
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
# 4 half-neighbourhoods: each undirected edge is generated only once
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


# =============================================================================
# Step 1: cells of the domain
# =============================================================================
def centers(grid, rows, cols) -> np.ndarray:
    return shapely.points(grid.xmin + (cols + 0.5) * grid.res,
                          grid.ymax - (rows + 0.5) * grid.res)


def burn(geoms, grid) -> np.ndarray:
    """Boolean mask, pixel-centre rule (like coverage.weighted_cells)."""
    return rasterize(((g, 1) for g in geoms), out_shape=(grid.height, grid.width),
                     transform=from_origin(grid.xmin, grid.ymax, grid.res, grid.res),
                     fill=0, dtype="uint8").astype(bool)


def domain_cells(s, per, assets, cells_w3, cfg):
    g, ic = s.grid, cfg["intruder"]
    b, d_t = float(ic["outer_band_m"]), float(ic["target_contact_m"])
    fence = per.boundary

    inside = burn([per], g)
    asset = burn(assets.geometry, g)
    ground = s.z_opaque < s.z_tgt                     # an intruder can stand there (NaN -> False)

    # Interior: exactly the cells of week 3, in the same order
    ri, ci = cells_w3["rows"], cells_w3["cols"]
    assert inside[ri, ci].all() and ground[ri, ci].all() and not asset[ri, ci].any(), \
        "cells.npz inconsistent with the perimeter, the ground or the assets"
    missed = inside & ground & ~asset
    missed[ri, ci] = False
    n_missed = int(missed.sum())                      # interior ground outside the zones: obstacle

    # Exterior: ground outside the perimeter, closer than b to the fence
    band = burn([per.buffer(b + g.res)], g) & ~inside & ground
    re_, ce_ = np.nonzero(band)
    d_ext = shapely.distance(centers(g, re_, ce_), fence)
    keep = d_ext <= b
    re_, ce_, d_ext = re_[keep], ce_[keep], d_ext[keep]

    rows = np.concatenate([ri, re_])
    cols = np.concatenate([ci, ce_])
    interior = np.r_[np.ones(ri.size, bool), np.zeros(re_.size, bool)]
    pts = centers(g, rows, cols)
    dist = np.r_[-shapely.distance(pts[:ri.size], fence), d_ext]   # < 0 inside

    target = assets.loc[assets["intruder_target"].astype(bool)]
    assert len(target) == 1, f"{len(target)} intruder_target assets, 1 expected"
    tgt = target.geometry.iloc[0]
    is_target = interior & (shapely.distance(pts, tgt) <= d_t)     # never an exterior cell
    source = ~interior & (dist >= b - 1.5 * g.res)
    return dict(rows=rows, cols=cols, inside=interior, fence_dist=dist.astype(np.float32),
                source=source, target=is_target, target_name=target["name"].iloc[0],
                n_missed=n_missed, n_ext_near_target=int(((~interior) &
                                  (shapely.distance(pts, tgt) <= d_t)).sum()))


# =============================================================================
# Step 2: 8-connected graph
# =============================================================================
def build_edges(grid, rows, cols, interior):
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
            # diagonal forbidden if both orthogonal neighbours are obstacles
            o1 = idx[r0 + dr:r1 + dr, c0:c1]
            o2 = idx[r0:r1, c0 + dc:c1 + dc]
            ok &= (o1 >= 0) | (o2 >= 0)
        U.append(a[ok]); V.append(bb[ok])
        L.append(np.full(ok.sum(), grid.res * (SQRT2 if dr and dc else 1.0)))
    u, v, l = np.concatenate(U), np.concatenate(V), np.concatenate(L)
    return u, v, l, interior[u] != interior[v]


# =============================================================================
# Step 3: gates: opening = fence segment centred on the access point
# =============================================================================
def gate_width(type_str: str) -> float:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\b", str(type_str))
    if m is None:
        raise ValueError(f"width missing from the gate type '{type_str}' (e.g. 'Gate 3m')")
    return float(m.group(1).replace(",", "."))


def gate_edges(grid, rows, cols, u, v, crosses, per, access):
    fence = per.boundary
    gate = np.full(u.size, -1, np.int16)
    ix = np.flatnonzero(crosses)
    p = centers(grid, rows, cols)
    xy = shapely.get_coordinates(p)
    seg = shapely.linestrings(np.stack([xy[u[ix]], xy[v[ix]]], axis=1))
    table = []
    for k, a in enumerate(access.itertuples()):
        assert a.geometry.distance(fence) < 0.5, f"gate {a.name} off the fence"
        w = gate_width(a.type)
        opening = fence.intersection(a.geometry.buffer(w / 2))
        hit = shapely.intersects(seg, opening)
        assert (gate[ix[hit]] == -1).all(), f"gate {a.name} overlaps another gate"
        gate[ix[hit]] = k
        if not hit.any():
            raise ValueError(f"gate {a.name}: no crossable edge (obstacle in front?)")
        table.append(dict(gate=k, name=a.name, type=a.type, width_m=w,
                          opening_m=round(opening.length, 2), n_edges=int(hit.sum())))
    return gate, pd.DataFrame(table)


# =============================================================================
# Step 4: detection matrix on the domain, matched to the week 3 configurations
# =============================================================================
KEY = ["cand", "h", "sensor", "azimuth", "tilt"]


def config_key(df: pd.DataFrame) -> pd.Series:
    k = df[KEY].copy()
    k["azimuth"] = k["azimuth"].fillna(-1.0)
    k["tilt"] = k["tilt"].round(6)
    return k.astype(str).agg("|".join, axis=1)


def match_w3(conf: pd.DataFrame, conf_w3: pd.DataFrame) -> np.ndarray:
    pos = pd.Series(np.arange(len(conf_w3)), index=config_key(conf_w3))
    assert pos.index.is_unique
    return config_key(conf).map(pos).fillna(-1).to_numpy(np.int64)


def check_non_regression(A_int, A_w3, config_w3, n_int) -> float:
    """A_intruder restricted to the interior cells must reproduce A row by row."""
    have = config_w3 >= 0
    assert np.isin(np.arange(A_w3.shape[0]), config_w3[have]).all(), \
        "week 3 configurations missing from A_intruder"
    sub = A_int[np.flatnonzero(have)][:, :n_int]
    ref = A_w3[config_w3[have]]
    diff = abs(sub - ref)
    return float(diff.max()) if diff.nnz else 0.0


# =============================================================================
# Step 5: check figure
# =============================================================================
INK, INK_2 = "#0b0b0b", "#52514e"
C_SRC, C_TGT, C_FENCE, C_GATE = "#1baf7a", "#eb6834", "#0b0b0b", "#4a3aa7"
C_INT, C_BAND, C_OBST = "#dfe9f7", "#f1f0ec", "#b9b7b0"


def plot_domain(grid, dom, u, v, gate, per, assets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    r, c = dom["rows"], dom["cols"]
    img = np.full((grid.height, grid.width), np.nan)
    img[r, c] = dom["inside"].astype(float)             # 0 = outer band, 1 = inside
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
    ax.scatter(X[dom["target"]], Y[dom["target"]], s=2, color=C_TGT, lw=0)
    gt = gate >= 0
    ax.scatter((X[u[gt]] + X[v[gt]]) / 2, (Y[u[gt]] + Y[v[gt]]) / 2, s=14, color=C_GATE, lw=0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(color=C_INT, label="Inside"),
                       Patch(color=C_BAND, label=f"Outer band ({dom['b']:g} m)"),
                       Patch(color=C_OBST, label="Assets (obstacles)"),
                       Line2D([], [], color=C_FENCE, lw=1.2, label="Fence"),
                       Line2D([], [], marker="o", ls="", color=C_SRC, label="Intruder start"),
                       Line2D([], [], marker="o", ls="", color=C_TGT,
                              label=f"Target ({dom['target_name']})"),
                       Line2D([], [], marker="o", ls="", color=C_GATE,
                              label="Crossings through a gate")],
              frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=4, labelcolor=INK_2)
    ax.set_title("Intruder domain", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Step 6: orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    ic = cfg["intruder"]
    out = ROOT / ic["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cov = ROOT / cfg["coverage"]["out_dir"]

    s = prepare_surfaces(cfg)
    g = s.grid
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    derived = ROOT / cfg["site"]["output_gpkg"]
    assets = gpd.read_file(derived, layer="assets")
    access = gpd.read_file(derived, layer="access")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"])
    cells_w3 = np.load(cov / "cells.npz")
    A_w3 = sparse.load_npz(cov / "A.npz").tocsr()
    conf_w3 = pd.read_csv(cov / "configs.csv")

    # --- Cells
    dom = domain_cells(s, per, assets, cells_w3, cfg)
    dom["b"] = float(ic["outer_band_m"])
    n_int = int(dom["inside"].sum())
    print(f"Domain: {dom['rows'].size:,} cells = {n_int:,} inside "
          f"+ {dom['rows'].size - n_int:,} outside (band of {dom['b']:g} m)")
    print(f"  sources: {dom['source'].sum():,}   targets ({dom['target_name']}): "
          f"{dom['target'].sum():,}")
    print(f"  exterior cells within {ic['target_contact_m']} m of the target, excluded from the "
          f"targets: {dom['n_ext_near_target']}")
    print(f"  interior ground outside the weighted zones (obstacles): {dom['n_missed']}")

    # --- Graph
    u, v, l, crosses = build_edges(g, dom["rows"], dom["cols"], dom["inside"])
    gate, gates = gate_edges(g, dom["rows"], dom["cols"], u, v, crosses, per, access)
    print(f"\nGraph: {u.size:,} edges, of which {crosses.sum():,} cross the fence "
          f"and {(gate >= 0).sum()} through a gate")
    print(gates.to_string(index=False))

    # --- Matrix
    t0 = time.time()
    w_dom = np.r_[cells_w3["w"], np.zeros(dom["rows"].size - n_int)]
    A, conf, n_empty, lost = build_matrix(s, cand, dom["rows"], dom["cols"], w_dom, cfg)
    conf["config_w3"] = match_w3(conf, conf_w3)
    err = check_non_regression(A, A_w3, conf["config_w3"].to_numpy(), n_int)
    print(f"\nA_intruder: {A.shape[0]:,} × {A.shape[1]:,}, {A.nnz:,} non-zeros "
          f"({time.time() - t0:.0f} s); {n_empty} configurations without any detection")
    new = conf["config_w3"] < 0
    print(f"  week 3 configurations found: {(~new).sum():,} / {len(conf_w3):,}; "
          f"new ones (only see the outside): {new.sum():,}")
    print(f"  non-regression A_intruder[:, inside] vs A: max difference = {err:.2e}")
    assert err < 1e-6, "A_intruder does not reproduce A on the inside"
    print("  new configurations per sensor:",
          conf.loc[new, "sensor"].value_counts().to_dict())

    p_max = np.asarray(A.max(axis=0).todense()).ravel()
    ext = ~dom["inside"]
    near = ext & (dom["fence_dist"] <= 5)
    print(f"\nMax P (best single configuration) in the band: median "
          f"{np.median(p_max[ext]):.2f}, share < 0.5: {(p_max[ext] < 0.5).mean():.1%}; "
          f"within 5 m of the fence: share < 0.5: {(p_max[near] < 0.5).mean():.1%}")

    # --- Saving
    np.savez(out / "domain.npz", **{k: dom[k] for k in
             ("rows", "cols", "inside", "fence_dist", "source", "target")}, w=w_dom)
    np.savez(out / "graph.npz", u=u, v=v, length=l, crosses=crosses, gate=gate)
    sparse.save_npz(out / "A_intruder.npz", A)
    conf.to_csv(out / "intruder_configs.csv", index=False)
    gates.to_csv(out / "gates.csv", index=False)
    plot_domain(g, dom, u, v, gate, per, assets, out / "domain.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
