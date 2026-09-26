"""
src/opportune.py — Détection opportune (EASI simplifié, docs/formulation.md §9.1).

Une détection en v n'est utile que si l'intrus met encore au moins T_R secondes (temps de
réponse) à atteindre la cible. On note d(v) le temps minimal de v à la cible :

  d(v) = min_{γ : v → V_t} Σ_{e ∈ γ} (ℓ_e / vitesse + t_×(e))

d ne dépend pas des capteurs : un seul Dijkstra multi-source depuis les cibles, par profil.
Détection opportune masquée :  P̃_v = P_v · 1[d(v) ≥ T_R],  σ̃_v = σ_v · 1[d(v) ≥ T_R].

  p̃* = p*(P̃) minore la détection opportune garantie : le long d'un chemin, le temps qui
        reste réellement en u est ≥ d(u), donc tout nœud masqué-utile l'est réellement.
  Borne supérieure : détection opportune réelle (temps restant mesuré le long du chemin)
        sur deux couloirs réalisables de G_{p̃*} (le plus court en mètres, le plus rapide).
  Proposition : si T_R > T_min = min_{s ∈ V_s} d(s), alors p̃* = 0 pour tout dispositif.

Sorties : intrus/temps_restant.npz, intrus/opportune.csv, intrus/opportune_stats.csv,
          figures/temps_restant.png, figures/opportune.png
Usage   : python -m src.opportune
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from src.intrus import (PROFILS, Domaine, _plus_court, adjacency, detection, exposition,
                        franchissement, seuil_couloir, temps_aretes)


# =============================================================================
# Étape 1 — Temps restant d(v), indépendant des capteurs
# =============================================================================
def temps_restant(dom: Domaine, ic: dict, profil: str) -> np.ndarray:
    """d(v) en secondes (inf si la cible est inatteignable depuis v pour ce profil)."""
    mask = dom.aretes(profil)
    M = adjacency(dom, mask, temps_aretes(dom, ic)[mask])
    return dijkstra(M, directed=False, indices=np.flatnonzero(dom.cible), min_only=True)


def chemin_rapide(dom: Domaine, ic: dict, profil: str, node_ok=None) -> np.ndarray:
    """Chemin le plus rapide des départs à la cible (restreint aux nœuds node_ok)."""
    mask = dom.aretes(profil)
    if node_ok is not None:
        mask = mask & node_ok[dom.u] & node_ok[dom.v]
    return _plus_court(dom, mask, temps_aretes(dom, ic)[mask], node_ok)


# =============================================================================
# Étape 2 — Masque opportun et valeur réelle le long d'un chemin
# =============================================================================
def masquer(P: np.ndarray, sigma: np.ndarray, d: np.ndarray, T_R: float):
    utile = d >= T_R
    return np.where(utile, P, 0.0), np.where(utile, sigma, 0.0)


def reste_chemin(dom: Domaine, ic: dict, profil: str, chemin: np.ndarray) -> np.ndarray:
    """reste[k] = temps réel de chemin[k] jusqu'au bout du chemin (0 au dernier nœud)."""
    if chemin.size == 0:
        return np.zeros(0)
    mask = dom.aretes(profil)
    Tm = adjacency(dom, mask, temps_aretes(dom, ic)[mask])
    t = np.asarray(Tm[chemin[:-1], chemin[1:]]).ravel()
    assert (t > 0).all(), "arête absente du profil sur le chemin"
    return np.r_[np.cumsum(t[::-1])[::-1], 0.0]


def valeur_chemin(P: np.ndarray, chemin: np.ndarray, reste: np.ndarray, T_R: float) -> float:
    """Détection opportune réelle au meilleur point du chemin : max P sur les nœuds où reste ≥ T_R."""
    utile = reste >= T_R
    return float(P[chemin[utile]].max()) if utile.any() else 0.0


# =============================================================================
# Étape 3 — Évaluation d'un dispositif pour un temps de réponse
# =============================================================================
def evaluer_opportune(dom: Domaine, P, sigma, ic: dict, T_R: float, d: dict) -> dict:
    """p̃*, borne supérieure et P_chemin opportun (τ_obs nominal), pour F et G."""
    res = {}
    for pr in PROFILS:
        Pm, sm = masquer(P, sigma, d[pr], T_R)
        p, couloir = seuil_couloir(dom, Pm, pr)
        rapide = chemin_rapide(dom, ic, pr, Pm <= p)
        ub = min(valeur_chemin(P, c, reste_chemin(dom, ic, pr, c), T_R)
                 for c in (couloir, rapide) if c.size)
        assert p <= ub + 1e-12, "borne inférieure > borne supérieure"
        lam, _ = exposition(dom, sm, pr, ic, ic["tau_obs_s"])
        res[f"p_tilde_{pr}"] = p
        res[f"p_tilde_ub_{pr}"] = ub
        res[f"P_chemin_opp_{pr}"] = float(max(-np.expm1(-lam), 0.0))
    return res


def plafond(A) -> tuple[np.ndarray, np.ndarray]:
    """P et σ avec toutes les configurations actives : majorant de tout dispositif (P croît avec T)."""
    L = A.tocsc(copy=True).astype(np.float64)
    with np.errstate(divide="ignore"):
        L.data = -np.log1p(-L.data)
    sigma = np.asarray(L.sum(axis=0)).ravel()
    return np.maximum(-np.expm1(-sigma), 0.0), sigma


# =============================================================================
# Étape 4 — Solutions de référence : front NSGA-II de la semaine 4
# =============================================================================
def solutions_front(path, budgets) -> dict[float, tuple[np.ndarray, pd.Series]]:
    """Pour chaque B : solution de coût ≤ B de meilleur p*_F, puis de meilleur F."""
    f = pd.read_csv(path)
    out = {}
    for B in budgets:
        s = f[f["cout"] <= B + 1e-9].sort_values(["p_star_F", "F"], ascending=False).iloc[0]
        out[float(B)] = (np.array(str(s["configs"]).split(), dtype=np.int64), s)
    return out


# =============================================================================
# Étape 5 — Figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
C_F, C_G = "#eb6834", "#4a3aa7"
C_SOL = ("#2a78d6", "#1f9e6e", "#b8456a")


def _cadre(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_temps_restant(dom, grid, per, assets, d, rapides, niveaux, path):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, c = dom.rows, dom.cols
    r0, r1, c0, c1 = r.min() - 3, r.max() + 4, c.min() - 3, c.max() + 4
    ext = (grid.xmin + c0 * grid.res, grid.xmin + c1 * grid.res,
           grid.ymax - r1 * grid.res, grid.ymax - r0 * grid.res)
    X = grid.xmin + (c + 0.5) * grid.res
    Y = grid.ymax - (r + 0.5) * grid.res
    xs = grid.xmin + (np.arange(c0, c1) + 0.5) * grid.res
    ys = grid.ymax - (np.arange(r0, r1) + 0.5) * grid.res

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), dpi=150, layout="constrained")
    vmax = max(np.nanmax(np.where(np.isfinite(d[p]), d[p], np.nan)) for p in PROFILS)
    for ax, pr, col in zip(axes, PROFILS, (C_F, C_G)):
        img = np.full((grid.height, grid.width), np.nan)
        img[r, c] = np.where(np.isfinite(d[pr]), d[pr], np.nan)
        sub = img[r0:r1, c0:c1]
        im = ax.imshow(sub, extent=ext, cmap="viridis_r", vmin=0, vmax=vmax,
                       interpolation="nearest")
        cs = ax.contour(xs, ys, np.ma.masked_invalid(sub), levels=niveaux, colors="white",
                        linewidths=0.7)
        ax.clabel(cs, fmt=lambda v: f"{v:g} s", fontsize=7, colors="white")
        gpd.GeoSeries(list(assets.geometry)).plot(ax=ax, color="#d6d4cf", edgecolor=INK_2, lw=0.4)
        gpd.GeoSeries([per.boundary]).plot(ax=ax, color=INK, lw=1)
        p = rapides[pr]
        ax.plot(X[p], Y[p], color="white", lw=3.2)
        ax.plot(X[p], Y[p], color=col, lw=1.8)
        nom = "déterminé (F)" if pr == "F" else "opportuniste (G)"
        ax.set_title(f"Intrus {nom}\nT_min = {d[pr][dom.source].min():.1f} s",
                     loc="left", fontsize=10, color=INK)
        _cadre(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Temps restant jusqu'à la cible d(v) (s)", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_courbes(df, T_min, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=150, sharey=True,
                             layout="constrained")
    for ax, pr in zip(axes, PROFILS):
        for col, (B, g) in zip(C_SOL + ("#8a8883",), df.groupby("budget")):
            lab = "catalogue complet (plafond)" if not np.isfinite(B) else f"B = {B:g}"
            ax.plot(g["T_R"], g[f"p_tilde_{pr}"], color=col, lw=2, label=lab,
                    zorder=1 if not np.isfinite(B) else 2)
            if np.isfinite(B):
                ax.plot(g["T_R"], g[f"p_tilde_ub_{pr}"], color=col, lw=1, ls="--")
        ax.axvline(T_min[pr], color=INK_2, lw=0.8, ls=":")
        ax.text(T_min[pr], 1.02, f" T_min = {T_min[pr]:.1f} s", color=INK_2, fontsize=8,
                va="bottom")
        nom = "déterminé (F)" if pr == "F" else "opportuniste (G)"
        ax.set_title(f"Intrus {nom}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Temps de réponse T_R (s)", color=INK_2)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 60)                 # au-delà : p̃* = 0 partout (proposition §9.1)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Détection opportune garantie p̃*", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, title="trait plein : p̃*, tirets : borne sup.",
                   title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Étape 6 — Programme principal
# =============================================================================
def main() -> None:
    import geopandas as gpd

    from src.intrus import load_domaine
    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    ic = cfg["intrus"]
    out = ROOT / ic["out_dir"]
    fig_dir = ROOT / cfg["optimisation"]["figures_dir"]
    dom, A, conf = load_domaine(cfg, ROOT)

    # 1. Temps restant et temps d'intrusion minimal
    d = {pr: temps_restant(dom, ic, pr) for pr in PROFILS}
    assert (d["F"] <= d["G"] + 1e-9).all(), "violation de d_F ≤ d_G"
    np.savez_compressed(out / "temps_restant.npz", d_F=d["F"], d_G=d["G"])
    stats, rapides, T_min = [], {}, {}
    for pr in PROFILS:
        ds = d[pr][dom.source]
        ds = ds[np.isfinite(ds)]
        ch = chemin_rapide(dom, ic, pr)
        rapides[pr], T_min[pr] = ch, float(ds.min())
        noeud, gate = franchissement(dom, ch)
        k = int(np.flatnonzero(ch == noeud)[0]) if noeud >= 0 else -1
        reste = reste_chemin(dom, ic, pr, ch)
        stats.append(dict(profil=pr, T_min=T_min[pr], mediane=float(np.median(ds)),
                          p90=float(np.percentile(ds, 90)), T_max=float(ds.max()),
                          longueur_m=float(np.sum(np.hypot(np.diff(dom.rows[ch]),
                                                           np.diff(dom.cols[ch])))),
                          portail=gate,
                          temps_exterieur_s=float(reste[0] - reste[k]) if k >= 0 else np.nan,
                          temps_interieur_s=float(reste[k]) if k >= 0 else np.nan))
    st = pd.DataFrame(stats)
    st.to_csv(out / "opportune_stats.csv", index=False)
    print("Temps d'intrusion depuis les départs (s)")
    print(st.round(2).to_string(index=False))

    # 2. Solutions du front face à T_R
    budgets = (13, 20, 30)
    sols = solutions_front(ROOT / cfg["nsga"]["out_dir"] / "front_global.csv", budgets)
    grille = np.union1d(np.arange(0.0, 121.0, 2.5), ic["t_reponse_s"])
    rows = []
    cas = [(B, s["cout"], s["F"], detection(A, T)) for B, (T, s) in sols.items()]
    cas.append((np.inf, np.nan, np.nan, plafond(A)))          # catalogue complet : plafond
    for B, cout, F, (P, sigma) in cas:
        for T_R in grille:
            r = evaluer_opportune(dom, P, sigma, ic, float(T_R), d)
            rows.append(dict(budget=B, cout=cout, F=F, T_R=float(T_R), **r))
    df = pd.DataFrame(rows)
    for pr in PROFILS:             # proposition §9.1 : T_R > T_min ⇒ p̃* = 0
        assert (df.loc[df["T_R"] > T_min[pr], f"p_tilde_{pr}"] == 0).all()
        r0 = df[(df["T_R"] == 0) & np.isfinite(df["budget"])]   # T_R = 0 redonne p* (sem. 4)
        ref = np.array([sols[B][1][f"p_star_{pr}"] for B in r0["budget"]])
        assert np.allclose(r0[f"p_tilde_{pr}"], ref, atol=1e-6), "T_R = 0 ≠ p* du front"
    df.to_csv(out / "opportune.csv", index=False)

    tab = df[df["T_R"].isin(ic["t_reponse_s"])]
    cols = ["budget", "T_R"] + [f"{m}_{pr}" for pr in PROFILS
                                for m in ("p_tilde", "p_tilde_ub", "P_chemin_opp")]
    pd.set_option("display.width", 200)
    print("\nDétection opportune des solutions du front (meilleur p*_F à coût ≤ B)")
    print(tab[cols].round(3).to_string(index=False))

    # 3. Figures
    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs")
    niveaux = [t for t in ic["t_reponse_s"] if t > 0]
    plot_temps_restant(dom, g, per, assets, d, rapides, niveaux, fig_dir / "temps_restant.png")
    plot_courbes(df, T_min, fig_dir / "opportune.png")
    print(f"\n-> {fig_dir / 'temps_restant.png'}\n-> {fig_dir / 'opportune.png'}")


if __name__ == "__main__":
    main()
