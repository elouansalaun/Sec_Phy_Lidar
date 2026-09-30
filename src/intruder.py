"""
src/intruder.py: evaluation of a sensor layout by the intruder (docs/formulation.md, §7.3-7.6).

Two metrics on the graph produced by src.intruder_domain:

  p*  corridor threshold: min_γ max_{v∈γ} P_v. Guaranteed lower bound on the detection of the
      best path, whatever the correlation along the path (§7.4).
      Exact computation by union-find: nodes are added by increasing P until start and
      target are connected.
  Λ*  exposure: min_γ Σ c_uv, c_uv = (σ_u+σ_v)/2 · (ℓ_uv/v + t_×)/τ_obs, σ = −ln q (§7.5).
      P_path = 1 − exp(−Λ*). Multi-source Dijkstra.

Each one for two profiles: "F" (determined, crosses anywhere) and "G" (opportunistic,
gates only). Always m_F ≤ m_G (§7.3).

Usage : python -m src.intruder      (evaluates the MILP solutions of the analysis_budgets)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

PROFILES = ("F", "G")
EPS_W = 1e-12            # minimal edge weight: csgraph ignores explicit zeros


# =============================================================================
# Step 1: domain data
# =============================================================================
@dataclass(frozen=True)
class Domain:
    rows: np.ndarray
    cols: np.ndarray
    inside: np.ndarray         # (n,) bool
    source: np.ndarray         # (n,) bool
    target: np.ndarray         # (n,) bool
    u: np.ndarray              # (m,) undirected edges
    v: np.ndarray
    length: np.ndarray         # (m,) metres
    crosses: np.ndarray        # (m,) bool: the edge crosses the fence
    gate: np.ndarray           # (m,) index of the gate, −1 otherwise

    def __post_init__(self):
        n, m = self.rows.size, self.u.size
        assert self.inside.shape == self.source.shape == self.target.shape == (n,)
        assert self.v.shape == self.length.shape == self.crosses.shape == self.gate.shape == (m,)
        assert self.source.any() and self.target.any()
        assert not (self.gate >= 0)[~self.crosses].any(), "gate on a non-crossing edge"

    @property
    def n(self) -> int:
        return self.rows.size

    def edges(self, profile: str) -> np.ndarray:
        """Mask of the allowed edges: F = all, G = no crossing outside a gate."""
        if profile == "F":
            return np.ones(self.u.size, bool)
        if profile == "G":
            return ~self.crosses | (self.gate >= 0)
        raise ValueError(f"unknown profile: {profile}")


def load_domain(cfg: dict, root) -> tuple[Domain, sparse.csr_matrix, pd.DataFrame]:
    d = root / cfg["intruder"]["out_dir"]
    dm, gr = np.load(d / "domain.npz"), np.load(d / "graph.npz")
    dom = Domain(rows=dm["rows"], cols=dm["cols"], inside=dm["inside"],
                 source=dm["source"], target=dm["target"], u=gr["u"], v=gr["v"],
                 length=gr["length"], crosses=gr["crosses"], gate=gr["gate"])
    A = sparse.load_npz(d / "A_intruder.npz").tocsr()
    conf = pd.read_csv(d / "intruder_configs.csv")
    assert A.shape == (len(conf), dom.n)
    return dom, A, conf


# =============================================================================
# Step 2: detection per node for a layout T
# =============================================================================
def detection(A: sparse.csr_matrix, T) -> tuple[np.ndarray, np.ndarray]:
    """P_v = 1 − Π(1 − P_vj) and σ_v = −ln q_v = −Σ ln(1 − P_vj) (inf if a sensor is certain)."""
    T = np.unique(np.asarray(T, dtype=np.int64))
    if T.size == 0:
        return np.zeros(A.shape[1]), np.zeros(A.shape[1])
    sub = A[T].toarray().astype(np.float64)
    with np.errstate(divide="ignore"):
        sigma = -np.log1p(-sub).sum(axis=0)
    return np.maximum(-np.expm1(-sigma), 0.0), sigma    # P = 1 − e^{−σ}, accurate near 0


# =============================================================================
# Step 3: corridor threshold p*: union-find by increasing P
# =============================================================================
@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _percolation(order, indptr, indices, is_src, is_tgt):
    """Adds the nodes in order; returns the position of the node that connects source and target."""
    n = is_src.size
    S, Tg = n, n + 1                                     # super-source, super-target
    parent = np.arange(n + 2)
    active = np.zeros(n, np.bool_)
    for k in range(order.size):
        x = order[k]
        active[x] = True
        if is_src[x]:
            a, b = _find(parent, x), _find(parent, S)
            if a != b:
                parent[a] = b
        if is_tgt[x]:
            a, b = _find(parent, x), _find(parent, Tg)
            if a != b:
                parent[a] = b
        for e in range(indptr[x], indptr[x + 1]):
            y = indices[e]
            if active[y]:
                a, b = _find(parent, x), _find(parent, y)
                if a != b:
                    parent[a] = b
        if _find(parent, S) == _find(parent, Tg):
            return k
    return -1


def adjacency(dom: Domain, mask: np.ndarray, weights=None) -> sparse.csr_matrix:
    u, v = dom.u[mask], dom.v[mask]
    w = np.ones(u.size) if weights is None else weights
    M = sparse.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(dom.n, dom.n))
    return M.tocsr()


def corridor_threshold(dom: Domain, P: np.ndarray, profile: str) -> tuple[float, np.ndarray]:
    """p* and a corridor achieving p* (the shortest in metres among those of G_{p*})."""
    mask = dom.edges(profile)
    M = adjacency(dom, mask)
    order = np.argsort(P, kind="stable")
    k = _percolation(order, M.indptr, M.indices, dom.source, dom.target)
    if k < 0:
        raise ValueError(f"profile {profile}: no target reachable from the starts")
    p_star = float(P[order[k]])
    ok = P <= p_star
    keep = mask & ok[dom.u] & ok[dom.v]
    return p_star, _shortest(dom, keep, dom.length[keep], ok)


# =============================================================================
# Step 4: exposure Λ*: multi-source Dijkstra
# =============================================================================
def edge_times(dom: Domain, ic: dict) -> np.ndarray:
    """Travel time of each edge (s): walking ℓ/v + crossing time t_×."""
    t_x = np.where(dom.gate >= 0, ic["t_gate_s"],
                   np.where(dom.crosses, ic["t_fence_s"], 0.0))
    return dom.length / ic["speed_m_s"] + t_x


def exposure_weights(dom: Domain, sigma: np.ndarray, ic: dict, tau_obs: float) -> np.ndarray:
    return 0.5 * (sigma[dom.u] + sigma[dom.v]) * edge_times(dom, ic) / tau_obs


def exposure(dom: Domain, sigma: np.ndarray, profile: str, ic: dict,
             tau_obs: float) -> tuple[float, np.ndarray]:
    """Λ* (inf if every path goes through a cell with certain detection) and the path."""
    w = exposure_weights(dom, sigma, ic, tau_obs)
    keep = dom.edges(profile) & np.isfinite(w)
    return _shortest(dom, keep, w[keep] + EPS_W, None, value=True)


def _shortest(dom, keep, w, node_ok, value=False):
    M = adjacency(dom, keep, w)
    src = np.flatnonzero(dom.source if node_ok is None else dom.source & node_ok)
    dist, pred, _ = dijkstra(M, directed=False, indices=src, min_only=True,
                             return_predecessors=True)
    tg = np.flatnonzero(dom.target)
    end = tg[np.argmin(dist[tg])]
    if not np.isfinite(dist[end]):
        return (np.inf, np.array([], np.int64)) if value else np.array([], np.int64)
    path = [end]
    while pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    path = np.array(path[::-1], np.int64)
    if value:
        return float(dist[end] - EPS_W * (path.size - 1)), path
    return path


# =============================================================================
# Step 5: full evaluation of a layout
# =============================================================================
def crossing(dom: Domain, path: np.ndarray) -> tuple[int, int]:
    """(index of the interior node at the crossing, gate or −1); (−1, −1) if none."""
    side = dom.inside[path]
    k = np.flatnonzero(side[1:] != side[:-1])
    if k.size == 0:
        return -1, -1
    a, b = path[k[0]], path[k[0] + 1]
    e = np.flatnonzero(((dom.u == a) & (dom.v == b)) | ((dom.u == b) & (dom.v == a)))
    return int(b if dom.inside[b] else a), int(dom.gate[e[0]])


def evaluate_intruder(dom: Domain, A, T, ic: dict, paths: bool = False) -> dict:
    P, sigma = detection(A, T)
    res, found = {}, {}
    for pr in PROFILES:
        p, corridor = corridor_threshold(dom, P, pr)
        res[f"p_star_{pr}"] = p
        found[f"corridor_{pr}"] = corridor
        for tau in ic["tau_obs_sensitivity_s"]:
            lam, ch = exposure(dom, sigma, pr, ic, tau)
            res[f"P_path_{pr}_tau{tau:g}"] = float(max(-np.expm1(-lam), 0.0))
            if tau == ic["tau_obs_s"]:
                res[f"Lambda_{pr}"] = lam
                found[f"path_{pr}"] = ch
        _, gate = crossing(dom, corridor)
        res[f"gate_corridor_{pr}"] = gate
        res[f"corridor_length_{pr}"] = float(len(corridor))
    pi = ic["pi_gate"]
    res["D_pi"] = pi * res["p_star_G"] + (1 - pi) * res["p_star_F"]
    assert res["p_star_F"] <= res["p_star_G"] + 1e-12, "violation of m_F ≤ m_G"
    return (res, P, found) if paths else res


# =============================================================================
# Step 6: demonstration: MILP solutions of week 3
# =============================================================================
def solutions_w3(conf: pd.DataFrame, path) -> dict[float, np.ndarray]:
    """Solutions indexed in configs.csv -> indices of A_intruder (column config_w3)."""
    to_new = pd.Series(np.arange(len(conf)), index=conf["config_w3"])
    sols = pd.read_csv(path)
    return {float(B): to_new.loc[g["config"]].to_numpy() for B, g in sols.groupby("budget")}


def plot_paths(dom, grid, per, assets, cases, path):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    INK, INK_2 = "#0b0b0b", "#52514e"
    C_F, C_G = "#eb6834", "#4a3aa7"
    r, c = dom.rows, dom.cols
    r0, r1, c0, c1 = r.min() - 3, r.max() + 4, c.min() - 3, c.max() + 4
    ext = (grid.xmin + c0 * grid.res, grid.xmin + c1 * grid.res,
           grid.ymax - r1 * grid.res, grid.ymax - r0 * grid.res)
    X = grid.xmin + (c + 0.5) * grid.res
    Y = grid.ymax - (r + 0.5) * grid.res

    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.6), dpi=150,
                             layout="constrained")
    for ax, (title, res, P, paths) in zip(np.atleast_1d(axes), cases):
        img = np.full((grid.height, grid.width), np.nan)
        img[r, c] = P
        im = ax.imshow(img[r0:r1, c0:c1], extent=ext, cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")
        gpd.GeoSeries(list(assets.geometry)).plot(ax=ax, color="#d6d4cf", edgecolor=INK_2, lw=0.4)
        gpd.GeoSeries([per.boundary]).plot(ax=ax, color=INK, lw=1)
        for key, col in (("corridor_G", C_G), ("corridor_F", C_F)):
            p = paths[key]
            ax.plot(X[p], Y[p], color="white", lw=3.2)
            ax.plot(X[p], Y[p], color=col, lw=1.8)
        ax.set_title(f"{title}\np*_F = {res['p_star_F']:.2f}   p*_G = {res['p_star_G']:.2f}",
                     loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Detection per cell P", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.legend(handles=[Line2D([], [], color=C_F, lw=2, label="Corridor, determined intruder (F)"),
                        Line2D([], [], color=C_G, lw=2, label="Corridor, opportunistic intruder (G)")],
               frameon=False, fontsize=9, loc="outside lower center", ncol=2, labelcolor=INK_2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import geopandas as gpd

    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    ic = cfg["intruder"]
    out = ROOT / ic["out_dir"]
    dom, A, conf = load_domain(cfg, ROOT)
    sols = solutions_w3(conf, ROOT / cfg["optimization"]["out_dir"] / "milp_solutions.csv")
    budgets = cfg["optimization"]["analysis_budgets"]
    w = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")["w"]

    rows, cases = [], []
    for B in budgets:
        res, P, paths = evaluate_intruder(dom, A, sols[float(B)], ic, paths=True)
        F = float(w @ P[dom.inside]) / float(w.sum())
        rows.append(dict(budget=B, F=F, **res))
        cases.append((f"MILP, B = {B:g}  (F = {100 * F:.0f} %)", res, P, paths))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.T.round(3).to_string(header=False))

    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets")
    plot_paths(dom, g, per, assets, cases, out / "milp_paths.png")
    print(f"\n-> {out / 'milp_paths.png'}")


if __name__ == "__main__":
    main()
