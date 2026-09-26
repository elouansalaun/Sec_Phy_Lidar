"""
src/sensibilite.py — Analyse de sensibilité par regret de décision (docs/formulation.md §9.3).

Protocole préenregistré (bloc `sensibilite` de config.yaml) : on reprend les décisions
nominales D_0 et D_10 (optima certifiés de src.boucle, θ = 0,9, T_R = 0 ou 10 s, fusion max)
et on mesure, pour chaque scénario s (un facteur à la fois) :

  tient      p̃*_s(D) ≥ θ
  c*_s       coût minimal certifié sous s (même boucle)
  regret     coût(D) − c*_s si D tient (sur-dimensionnement)
  surcoût    c*_s − coût(D) si D ne tient pas (défaillance)
  Jaccard    entre D et D*_s, configurations identifiées par leur clé

Facteurs : retard (t_c, t_p) et vitesse (seul d change) ; bande b et feuillage f
(domaine et A_intrus reconstruits par les fonctions de src.intrusion et src.coverage ;
k × f ⇔ α × f car τ est linéaire en k).
Frontière retard–réponse : T_min et c*(θ, T_R) pour chaque (t_c, t_p).

Sorties : sensibilite/regret.csv, sensibilite/frontiere.csv,
          figures/frontiere_retard.png, figures/regret.png
Usage   : python -m src.sensibilite            (≈ 1 à 2 h sur 2 cœurs)
          python -m src.sensibilite figures    (figures depuis les CSV)
"""
from __future__ import annotations

import copy
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from src.boucle import boucle, front_nsga, p_tilde
from src.instance import cost, from_configs
from src.intrus import Domaine, load_domaine
from src.intrusion import config_key
from src.opportune import temps_restant

PROFIL = "F"


# =============================================================================
# Étape 1 — Scénarios : reconstruction du domaine et de A_intrus
# =============================================================================
def construire(cfg: dict, root, bande: float, feuillage: float):
    """(Domaine, A, conf) pour une bande et un feuillage ; le nominal est relu sur disque."""
    ic = cfg["intrus"]
    if np.isclose(bande, ic["bande_ext_m"]) and np.isclose(feuillage, 1.0):
        return load_domaine(cfg, root)

    import geopandas as gpd

    from src.coverage import build_matrix
    from src.intrusion import build_edges, domain_cells, gate_edges
    from src.viewshed import prepare_surfaces

    c = copy.deepcopy(cfg)
    c["intrus"]["bande_ext_m"] = float(bande)
    for cap in c["sensors"]["cameras"] + c["sensors"]["radars"]:
        cap["alpha"] = float(cap["alpha"]) * float(feuillage)
    s = prepare_surfaces(c)
    per = gpd.read_file(root / c["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    der = root / c["site"]["output_gpkg"]
    assets = gpd.read_file(der, layer="actifs")
    acces = gpd.read_file(der, layer="acces")
    cand = gpd.read_file(root / c["candidates"]["output"])
    cells = np.load(root / c["coverage"]["out_dir"] / "cellules.npz")
    dm = domain_cells(s, per, assets, cells, c)
    u, v, l, fr = build_edges(s.grid, dm["rows"], dm["cols"], dm["interieur"])
    portail, _ = gate_edges(s.grid, dm["rows"], dm["cols"], u, v, fr, per, acces)
    w = np.r_[cells["w"], np.zeros(dm["rows"].size - int(dm["interieur"].sum()))]
    A, conf, *_ = build_matrix(s, cand, dm["rows"], dm["cols"], w, c)
    dom = Domaine(rows=dm["rows"], cols=dm["cols"], interieur=dm["interieur"],
                  source=dm["source"], cible=dm["cible"], u=u, v=v, longueur=l,
                  franchit=fr, portail=portail)
    return dom, A.tocsr(), conf


def cles(conf: pd.DataFrame) -> np.ndarray:
    """Clé (emplacement, hauteur, capteur, azimut, inclinaison), celle de src.intrusion."""
    return config_key(conf).to_numpy()


def traduire(T, cles_src: np.ndarray, index_dst: dict) -> tuple[np.ndarray, int]:
    """Indices dans le scénario ; les configurations absentes (aucune détection) sont comptées."""
    out = [index_dst[k] for k in cles_src[np.asarray(T, np.int64)] if k in index_dst]
    return np.array(sorted(out), np.int64), len(T) - len(out)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# =============================================================================
# Étape 2 — Évaluation d'un scénario
# =============================================================================
def evaluer(nom, facteur, valeur, cfg, dom, A, conf, ic_s, decisions, t_reponse, theta,
            graines_cles, reservoir, temps_max_s, verbose=False) -> list[dict]:
    """Une ligne par T_R : c*_s certifié, et regret de la décision nominale si elle existe."""
    from src.site import ROOT
    cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    n_int = int(dom.interieur.sum())
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    k_s = cles(conf)
    idx = {k: i for i, k in enumerate(k_s)}
    graines = [g for g in (traduire_cles(gk, idx) for gk in graines_cles) if g.size]
    d = temps_restant(dom, ic_s, PROFIL)
    T_min = float(d[dom.source].min())
    rows = []
    for T_R in t_reponse:
        base = dict(scenario=nom, facteur=facteur, valeur=valeur, T_R=T_R, T_min=T_min)
        D = decisions.get(T_R)
        if D is not None:
            D_s = np.array(sorted(idx[k] for k in D["cles"] if k in idx), np.int64)
            p_D = p_tilde(dom, A, D_s, d, T_R, PROFIL) if D_s.size else 0.0
            base.update(cout_D=D["cout"], p_D=p_D, tient=bool(p_D >= theta - 1e-9),
                        n_absentes=len(D["cles"]) - D_s.size)
        if T_R >= T_min:                       # proposition 2 : impossible, sans calcul
            rows.append(dict(base, statut="impossible (T_R ≥ T_min)", c_star=np.nan))
            continue
        t0 = time.perf_counter()
        r = boucle(inst, dom, A, d, theta, T_R, PROFIL, "cout",
                   graines=graines + ([D_s] if D is not None and D_s.size else []),
                   n_chemins=20, temps_max_s=temps_max_s, temps_maitre_s=300, fusion="max",
                   reservoir=reservoir, verbose=verbose)
        row = dict(base, statut=r.statut, c_star=r.valeur, borne=r.borne,
                   temps_s=time.perf_counter() - t0, iterations=r.iterations,
                   D_star=" ".join(k_s[r.T]) if r.T.size else "")
        if r.T.size:
            chk = p_tilde(dom, A, r.T, d, T_R, PROFIL)
            assert r.statut != "optimal" or chk >= theta - 1e-6, f"{nom} : p̃* = {chk}"
            kinds = conf.iloc[r.T]["capteur"].str.startswith("radar")
            row.update(n_radar=int(kinds.sum()), n_camera=int((~kinds).sum()))
        if D is not None and r.T.size:
            row["jaccard"] = jaccard(set(D["cles"]), set(k_s[r.T]))
            if row["tient"]:
                row["regret"] = D["cout"] - r.valeur
            else:
                row["surcout"] = r.valeur - D["cout"]
        rows.append(row)
        if True:                               # une ligne par résolution
            print(f"  {nom:<22} T_R = {T_R:>5g} s | T_min {T_min:5.1f} s | c* = {r.valeur:6.2f} "
                  f"({r.statut}, {row['temps_s']:.0f} s)"
                  + (f" | D : p̃* = {base['p_D']:.3f}, {'tient' if base['tient'] else 'NE TIENT PAS'}"
                     if D is not None else ""), flush=True)
    return rows


def traduire_cles(gk: np.ndarray, idx: dict) -> np.ndarray:
    return np.array(sorted(idx[k] for k in gk if k in idx), np.int64)


# =============================================================================
# Étape 3 — Figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
PAL = ("#2a78d6", "#1f9e6e", "#eb6834", "#b8456a", "#4a3aa7")
C_OK, C_KO = "#1f9e6e", "#d03b3b"


def plot_frontiere(fr: pd.DataFrame, path):
    """Matrice retard × temps de réponse : coût minimal certifié, ou impossibilité prouvée."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lignes = list(dict.fromkeys(fr["valeur"]))
    cols = sorted(fr["T_R"].unique())
    M = np.full((len(lignes), len(cols)), np.nan)
    tmin = {}
    for _, r in fr.iterrows():
        i, j = lignes.index(r["valeur"]), cols.index(r["T_R"])
        tmin[r["valeur"]] = r["T_min"]
        if r["statut"] == "optimal":
            M[i, j] = r["c_star"]
    cmap = LinearSegmentedColormap.from_list("c", ["#dcebfa", "#2a78d6"])
    cmap.set_bad("#f1efea")
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=150, layout="constrained")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=np.nanmin(M) - 0.5,
              vmax=np.nanmax(M) + 0.5, aspect="auto")
    for i in range(len(lignes)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, color=INK)
            else:
                ax.text(j, i, "impossible", ha="center", va="center", fontsize=7.5,
                        color="#8a8883", style="italic")
    ax.set_xticks(range(len(cols)), [f"{c:g} s" for c in cols])
    ax.set_yticks(range(len(lignes)),
                  [f"{v} s  (T_min = {tmin[v]:.0f} s)" for v in lignes])
    ax.set_xlabel("Temps de réponse T_R", color=INK_2)
    ax.set_ylabel("Retard (escalade, portail)", color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Coût minimal certifié pour une détection opportune garantie ≥ 0,9",
                 loc="left", fontsize=9, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_regret(rg: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decs = sorted(rg["T_R"].unique())
    fig, axes = plt.subplots(1, len(decs), figsize=(4.4 * len(decs), 4.2), dpi=150,
                             sharey=True, layout="constrained")
    for ax, T_R in zip(np.atleast_1d(axes), decs):
        g = rg[rg["T_R"] == T_R].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        cD = g["cout_D"].iloc[0]
        ax.axvline(cD, color=INK_2, lw=1, ls="--")
        for yi, (_, r) in zip(y, g.iterrows()):
            col = C_OK if r["tient"] else C_KO
            if np.isfinite(r["c_star"]):
                ax.plot([r["c_star"], cD], [yi, yi], color=col, lw=1.5, alpha=0.5)
                ax.plot(r["c_star"], yi, "o", color=col, ms=6)
            lab = f"p̃* = {r['p_D']:.2f}" + ("" if r["tient"] else "  ✗")
            x_txt = max(cD, r["c_star"]) if np.isfinite(r["c_star"]) else cD
            ax.text(x_txt + 0.3, yi, lab, va="center", fontsize=7, color=col)
        ax.set_yticks(y); ax.set_yticklabels(g["scenario"], fontsize=8)
        ax.set_title(f"Décision D_{T_R:g} (coût {cD:g}, tirets)", loc="left", fontsize=9,
                     color=INK)
        ax.set_xlabel("Coût minimal certifié sous le scénario c*_s", color=INK_2)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlim(right=ax.get_xlim()[1] + 3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Vert : la décision nominale tient (écart = regret) ; rouge : elle ne tient "
                 "pas (écart = surcoût nécessaire)", fontsize=8, color=INK_2, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figures(root, cfg):
    out = root / cfg["sensibilite"]["out_dir"]
    fig_dir = root / cfg["optimisation"]["figures_dir"]
    fr = pd.read_csv(out / "frontiere.csv")
    plot_frontiere(fr, fig_dir / "frontiere_retard.png")
    rg = pd.read_csv(out / "regret.csv")
    plot_regret(rg[rg["T_R"].isin(cfg["sensibilite"]["decisions_t_reponse_s"])],
                fig_dir / "regret.png")


# =============================================================================
# Étape 4 — Programme principal
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures(ROOT, cfg)
        return
    sc, ic = cfg["sensibilite"], cfg["intrus"]
    out = ROOT / sc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    theta = float(sc["theta"])

    # Décisions nominales et graines, identifiées par clé
    dom0, A0, conf0 = load_domaine(cfg, ROOT)
    k0 = cles(conf0)
    sol = pd.read_csv(ROOT / cfg["boucle"]["out_dir"] / "solutions.csv")
    cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    inst0 = from_configs(A0[:, :int(dom0.interieur.sum())], cells["w"], conf0, cfg)
    decisions = {}
    for T_R in sc["decisions_t_reponse_s"]:
        T = sol[(sol["fusion"] == "max") & np.isclose(sol["theta"], theta)
                & np.isclose(sol["T_R"], T_R)]["config"].to_numpy(np.int64)
        assert T.size, f"décision nominale absente pour T_R = {T_R} : lancer src.boucle"
        decisions[float(T_R)] = dict(cles=list(k0[T]), cout=cost(inst0, T))
        print(f"D_{T_R:g} : coût {decisions[float(T_R)]['cout']:g}, "
              f"{', '.join(k0[T])}")
    graines_cles = [k0[T] for T in front_nsga(ROOT, cfg)]
    graines_cles += [np.array(D["cles"]) for D in decisions.values()]
    t_dec = sorted(decisions)
    rows, front = [], []

    # 1. Retard et vitesse : même domaine, seul d change ; un réservoir commun
    res0: list = []
    for tc, tp in sc["retard"]:
        ic_s = dict(ic, t_cloture_s=float(tc), t_portail_s=float(tp))
        nom = f"retard ({tc:g}, {tp:g}) s"
        t_all = sorted(set(t_dec) | set(map(float, sc["retard_t_reponse_s"])))
        r = evaluer(nom, "retard", f"({tc:g}, {tp:g})", cfg, dom0, A0, conf0, ic_s,
                    decisions, t_all, theta, graines_cles, res0, sc["temps_max_s"])
        front += r
        rows += [x for x in r if x["T_R"] in t_dec]
    for vit in sc["vitesse_m_s"]:
        ic_s = dict(ic, vitesse_m_s=float(vit))
        rows += evaluer(f"vitesse {vit:g} m/s", "vitesse", vit, cfg, dom0, A0, conf0, ic_s,
                        decisions, t_dec, theta, graines_cles, res0, sc["temps_max_s"])
    pd.DataFrame(front).to_csv(out / "frontiere.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "regret.csv", index=False)

    # 2. Bande et feuillage : domaine et A_intrus reconstruits ; un réservoir par scénario
    for facteur, valeurs in (("bande", sc["bande_ext_m"]), ("feuillage", sc["feuillage"])):
        for val in valeurs:
            t0 = time.perf_counter()
            b = float(val) if facteur == "bande" else float(ic["bande_ext_m"])
            f = float(val) if facteur == "feuillage" else 1.0
            dom, A, conf = construire(cfg, ROOT, b, f)
            print(f"{facteur} = {val:g} : {dom.n:,} nœuds, {A.shape[0]:,} configurations, "
                  f"{A.nnz:,} non-nuls ({time.perf_counter() - t0:.0f} s)", flush=True)
            nom = f"bande {val:g} m" if facteur == "bande" else f"feuillage × {val:g}"
            rows += evaluer(nom, facteur, val, cfg, dom, A, conf, ic, decisions, t_dec, theta,
                            graines_cles, [], sc["temps_max_s"])
            pd.DataFrame(rows).to_csv(out / "regret.csv", index=False)

    rg = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    cols = ["scenario", "T_R", "T_min", "cout_D", "p_D", "tient", "c_star", "statut", "regret",
            "surcout", "jaccard", "n_radar", "n_camera"]
    print(rg[[c for c in cols if c in rg]].round(3).to_string(index=False))
    for T_R, g in rg.groupby("T_R"):
        D = decisions[T_R]
        stable = g["tient"].all() and (g["regret"].fillna(0) < sc["seuil_stable"] * D["cout"]).all()
        print(f"D_{T_R:g} : {'STABLE' if stable else 'NON STABLE'} "
              f"(tient dans {int(g['tient'].sum())}/{len(g)} scénarios)")
    figures(ROOT, cfg)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
