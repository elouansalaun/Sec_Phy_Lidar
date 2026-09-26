"""
src/intrus.py — Évaluation d'un dispositif par l'intrus (docs/formulation.md, §7.3–7.6).

Deux métriques sur le graphe produit par src.intrusion :

  p*  seuil du couloir : min_γ max_{v∈γ} P_v. Borne inférieure garantie de la détection
      du meilleur chemin, quelle que soit la corrélation le long du chemin (§7.4).
      Calcul exact par union-find : on ajoute les nœuds par P croissante jusqu'à ce que
      départ et cible soient reliés.
  Λ*  exposition : min_γ Σ c_uv, c_uv = (σ_u+σ_v)/2 · (ℓ_uv/v + t_×)/τ_obs, σ = −ln q (§7.5).
      P_chemin = 1 − exp(−Λ*). Dijkstra multi-source.

Chacune pour deux profils : « F » (déterminé, franchit partout) et « G » (opportuniste,
portails seulement). Toujours m_F ≤ m_G (§7.3).

Usage : python -m src.intrus      (évalue les solutions MILP des budgets de analyse_budgets)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

PROFILS = ("F", "G")
EPS_W = 1e-12            # poids minimal d'une arête : csgraph ignore les zéros explicites


# =============================================================================
# Étape 1 — Données du domaine
# =============================================================================
@dataclass(frozen=True)
class Domaine:
    rows: np.ndarray
    cols: np.ndarray
    interieur: np.ndarray      # (n,) bool
    source: np.ndarray         # (n,) bool
    cible: np.ndarray          # (n,) bool
    u: np.ndarray              # (m,) arêtes non orientées
    v: np.ndarray
    longueur: np.ndarray       # (m,) mètres
    franchit: np.ndarray       # (m,) bool : l'arête coupe la clôture
    portail: np.ndarray        # (m,) indice du portail, −1 sinon

    def __post_init__(self):
        n, m = self.rows.size, self.u.size
        assert self.interieur.shape == self.source.shape == self.cible.shape == (n,)
        assert self.v.shape == self.longueur.shape == self.franchit.shape == self.portail.shape == (m,)
        assert self.source.any() and self.cible.any()
        assert not (self.portail >= 0)[~self.franchit].any(), "portail sur une arête non franchissante"

    @property
    def n(self) -> int:
        return self.rows.size

    def aretes(self, profil: str) -> np.ndarray:
        """Masque des arêtes autorisées : F = toutes, G = pas de franchissement hors portail."""
        if profil == "F":
            return np.ones(self.u.size, bool)
        if profil == "G":
            return ~self.franchit | (self.portail >= 0)
        raise ValueError(f"profil inconnu : {profil}")


def load_domaine(cfg: dict, root) -> tuple[Domaine, sparse.csr_matrix, pd.DataFrame]:
    d = root / cfg["intrus"]["out_dir"]
    dm, gr = np.load(d / "domaine.npz"), np.load(d / "graphe.npz")
    dom = Domaine(rows=dm["rows"], cols=dm["cols"], interieur=dm["interieur"],
                  source=dm["source"], cible=dm["cible"], u=gr["u"], v=gr["v"],
                  longueur=gr["longueur"], franchit=gr["franchit"], portail=gr["portail"])
    A = sparse.load_npz(d / "A_intrus.npz").tocsr()
    conf = pd.read_csv(d / "configs_intrus.csv")
    assert A.shape == (len(conf), dom.n)
    return dom, A, conf


# =============================================================================
# Étape 2 — Détection par nœud pour un dispositif T
# =============================================================================
def detection(A: sparse.csr_matrix, T) -> tuple[np.ndarray, np.ndarray]:
    """P_v = 1 − Π(1 − P_vj) et σ_v = −ln q_v = −Σ ln(1 − P_vj) (inf si un capteur est certain)."""
    T = np.unique(np.asarray(T, dtype=np.int64))
    if T.size == 0:
        return np.zeros(A.shape[1]), np.zeros(A.shape[1])
    sub = A[T].toarray().astype(np.float64)
    with np.errstate(divide="ignore"):
        sigma = -np.log1p(-sub).sum(axis=0)
    return np.maximum(-np.expm1(-sigma), 0.0), sigma    # P = 1 − e^{−σ}, précis près de 0


# =============================================================================
# Étape 3 — Seuil du couloir p* : union-find par P croissante
# =============================================================================
@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _percolation(order, indptr, indices, is_src, is_tgt):
    """Ajoute les nœuds dans l'ordre ; renvoie la position du nœud qui relie source et cible."""
    n = is_src.size
    S, Tg = n, n + 1                                     # super-source, super-cible
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


def adjacency(dom: Domaine, mask: np.ndarray, weights=None) -> sparse.csr_matrix:
    u, v = dom.u[mask], dom.v[mask]
    w = np.ones(u.size) if weights is None else weights
    M = sparse.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(dom.n, dom.n))
    return M.tocsr()


def seuil_couloir(dom: Domaine, P: np.ndarray, profil: str) -> tuple[float, np.ndarray]:
    """p* et un couloir réalisant p* (le plus court en mètres parmi ceux de G_{p*})."""
    mask = dom.aretes(profil)
    M = adjacency(dom, mask)
    order = np.argsort(P, kind="stable")
    k = _percolation(order, M.indptr, M.indices, dom.source, dom.cible)
    if k < 0:
        raise ValueError(f"profil {profil} : aucune cible atteignable depuis les départs")
    p_star = float(P[order[k]])
    ok = P <= p_star
    keep = mask & ok[dom.u] & ok[dom.v]
    return p_star, _plus_court(dom, keep, dom.longueur[keep], ok)


# =============================================================================
# Étape 4 — Exposition Λ* : Dijkstra multi-source
# =============================================================================
def temps_aretes(dom: Domaine, ic: dict) -> np.ndarray:
    """Temps de parcours de chaque arête (s) : marche ℓ/v + temps de franchissement t_×."""
    t_x = np.where(dom.portail >= 0, ic["t_portail_s"],
                   np.where(dom.franchit, ic["t_cloture_s"], 0.0))
    return dom.longueur / ic["vitesse_m_s"] + t_x


def poids_exposition(dom: Domaine, sigma: np.ndarray, ic: dict, tau_obs: float) -> np.ndarray:
    return 0.5 * (sigma[dom.u] + sigma[dom.v]) * temps_aretes(dom, ic) / tau_obs


def exposition(dom: Domaine, sigma: np.ndarray, profil: str, ic: dict,
               tau_obs: float) -> tuple[float, np.ndarray]:
    """Λ* (inf si tout chemin traverse une cellule à détection certaine) et le chemin."""
    w = poids_exposition(dom, sigma, ic, tau_obs)
    keep = dom.aretes(profil) & np.isfinite(w)
    return _plus_court(dom, keep, w[keep] + EPS_W, None, value=True)


def _plus_court(dom, keep, w, node_ok, value=False):
    M = adjacency(dom, keep, w)
    src = np.flatnonzero(dom.source if node_ok is None else dom.source & node_ok)
    dist, pred, _ = dijkstra(M, directed=False, indices=src, min_only=True,
                             return_predecessors=True)
    tg = np.flatnonzero(dom.cible)
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
# Étape 5 — Évaluation complète d'un dispositif
# =============================================================================
def franchissement(dom: Domaine, path: np.ndarray) -> tuple[int, int]:
    """(indice du nœud intérieur au franchissement, portail ou −1) ; (−1, −1) si aucun."""
    side = dom.interieur[path]
    k = np.flatnonzero(side[1:] != side[:-1])
    if k.size == 0:
        return -1, -1
    a, b = path[k[0]], path[k[0] + 1]
    e = np.flatnonzero(((dom.u == a) & (dom.v == b)) | ((dom.u == b) & (dom.v == a)))
    return int(b if dom.interieur[b] else a), int(dom.portail[e[0]])


def evaluer(dom: Domaine, A, T, ic: dict, chemins: bool = False) -> dict:
    P, sigma = detection(A, T)
    res, paths = {}, {}
    for pr in PROFILS:
        p, couloir = seuil_couloir(dom, P, pr)
        res[f"p_star_{pr}"] = p
        paths[f"couloir_{pr}"] = couloir
        for tau in ic["tau_obs_sensibilite_s"]:
            lam, ch = exposition(dom, sigma, pr, ic, tau)
            res[f"P_chemin_{pr}_tau{tau:g}"] = float(max(-np.expm1(-lam), 0.0))
            if tau == ic["tau_obs_s"]:
                res[f"Lambda_{pr}"] = lam
                paths[f"chemin_{pr}"] = ch
        _, gate = franchissement(dom, couloir)
        res[f"portail_couloir_{pr}"] = gate
        res[f"longueur_couloir_{pr}"] = float(len(couloir))
    pi = ic["pi_portail"]
    res["D_pi"] = pi * res["p_star_G"] + (1 - pi) * res["p_star_F"]
    assert res["p_star_F"] <= res["p_star_G"] + 1e-12, "violation de m_F ≤ m_G"
    return (res, P, paths) if chemins else res


# =============================================================================
# Étape 6 — Démonstration : solutions MILP de la semaine 3
# =============================================================================
def solutions_s3(conf: pd.DataFrame, path) -> dict[float, np.ndarray]:
    """Solutions indexées dans configs.csv -> indices de A_intrus (colonne config_s3)."""
    to_new = pd.Series(np.arange(len(conf)), index=conf["config_s3"])
    sols = pd.read_csv(path)
    return {float(B): to_new.loc[g["config"]].to_numpy() for B, g in sols.groupby("budget")}


def plot_chemins(dom, grid, per, assets, cas, path):
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

    fig, axes = plt.subplots(1, len(cas), figsize=(5.2 * len(cas), 4.6), dpi=150,
                             layout="constrained")
    for ax, (titre, res, P, paths) in zip(np.atleast_1d(axes), cas):
        img = np.full((grid.height, grid.width), np.nan)
        img[r, c] = P
        im = ax.imshow(img[r0:r1, c0:c1], extent=ext, cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")
        gpd.GeoSeries(list(assets.geometry)).plot(ax=ax, color="#d6d4cf", edgecolor=INK_2, lw=0.4)
        gpd.GeoSeries([per.boundary]).plot(ax=ax, color=INK, lw=1)
        for key, col in (("couloir_G", C_G), ("couloir_F", C_F)):
            p = paths[key]
            ax.plot(X[p], Y[p], color="white", lw=3.2)
            ax.plot(X[p], Y[p], color=col, lw=1.8)
        ax.set_title(f"{titre}\np*_F = {res['p_star_F']:.2f}   p*_G = {res['p_star_G']:.2f}",
                     loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Détection par cellule P", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.legend(handles=[Line2D([], [], color=C_F, lw=2, label="Couloir, intrus déterminé (F)"),
                        Line2D([], [], color=C_G, lw=2, label="Couloir, intrus opportuniste (G)")],
               frameon=False, fontsize=9, loc="outside lower center", ncol=2, labelcolor=INK_2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import geopandas as gpd

    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    ic = cfg["intrus"]
    out = ROOT / ic["out_dir"]
    dom, A, conf = load_domaine(cfg, ROOT)
    sols = solutions_s3(conf, ROOT / cfg["optimisation"]["out_dir"] / "milp_solutions.csv")
    budgets = cfg["optimisation"]["analyse_budgets"]

    rows, cas = [], []
    for B in budgets:
        res, P, paths = evaluer(dom, A, sols[float(B)], ic, chemins=True)
        F = float(np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")["w"]
                  @ P[dom.interieur]) / float(np.load(ROOT / cfg["coverage"]["out_dir"]
                                                      / "cellules.npz")["w"].sum())
        rows.append(dict(budget=B, F=F, **res))
        cas.append((f"MILP, B = {B:g}  (F = {100 * F:.0f} %)", res, P, paths))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.T.round(3).to_string(header=False))

    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs")
    plot_chemins(dom, g, per, assets, cas, out / "chemins_milp.png")
    print(f"\n-> {out / 'chemins_milp.png'}")


if __name__ == "__main__":
    main()
