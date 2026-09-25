"""
src/catalogue_surrogate.py — Validation par la décision (§8.9, étape 6.3).

Question : un catalogue de positions candidates choisi par le modèle de substitution B2 donne-t-il
de meilleurs dispositifs que le catalogue fait à la main en semaine 1 ?

Six catalogues, aux MÊMES effectifs que celui de la semaine 1 (mâts et toits) :
  manuel          : data/site/candidats.gpkg (semaine 1)
  aleatoire       : positions possibles tirées au hasard (ce que vaut la sélection elle-même)
  B2, oracle      : suppression des non-maxima sur le potentiel GÉOMÉTRIQUE V̄, prédit par B2 ou
                    exact (6.3). Résultat : pire que le manuel, car V̄ ignore les zones pondérées.
  B2_pondere, oracle_pondere : idem sur le score PONDÉRÉ (6.3b) : chaque bin de V est pondéré par
                    la masse de poids w qu'il contient, calculée analytiquement ; le modèle de
                    substitution est inchangé, seule la connaissance du site s'ajoute.

Positions possibles (règles de la semaine 1), sur la grille du bloc test (pas de 2 m) :
  mâts  : sol à l'intérieur du périmètre, à ≥ `inset_m` de la clôture et ≥ `min_dist_building_m`
          du bâti ; hauteurs 4 à 10 m
  toits : cellules de toit à ≥ `inset_m` (toit) du bord du bâtiment ; hauteurs 2 et 4 m
Score géométrique d'une position : max, sur ses hauteurs possibles, du potentiel total V̄_h.
Score pondéré : avec M_b(o) = Σ_{j ∈ bin b} w_j,
  radar   R_h(o) = Σ_b M_b(o) · V_{h,b}(o)            (360°)
  caméra  C_h(o) = max_s Σ_a M_{a,s}(o) · V_{h,a,s}(o) (meilleur secteur)
  score(o) = max_h max(R_h / max R, C_h / max C)       (h parmi les hauteurs du support)
Hypothèse : dans un bin, visibilité et poids sont indépendants (V_b est une moyenne sur le bin).

Pour chaque catalogue, tout le reste est EXACT et identique : matrice sur le domaine de l'intrus
(coverage.build_matrix, comme src.intrusion), glouton sur la grille de budgets, p*_F et p*_G de
chaque solution, puis NSGA-II amorcé par les solutions gloutonnes du même catalogue.

Sorties (surrogate.out_dir/decision) : candidats_<nom>.gpkg, glouton_<nom>.csv, front_<nom>.csv,
                                       resume.csv ; figures : decision_budget.png, decision_front.png,
                                       decision_carte.png
Usage : python -m src.catalogue_surrogate                        (6 catalogues)
        python -m src.catalogue_surrogate B2_pondere oracle_pondere (sous-ensemble ; les autres
        sont relus dans decision/ s'ils ont déjà été évalués)
"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy import sparse

from src.baselines_surrogate import predire_b2
from src.candidates import sample
from src.coverage import build_matrix
from src.eval_surrogate import nms, potentiel_total, surfaces_bins
from src.greedy import greedy
from src.horizons import Horizons
from src.instance import KINDS, from_configs, violations
from src.intrus import Domaine, detection, seuil_couloir
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces

CATALOGUES = ("manuel", "aleatoire", "B2", "oracle", "B2_pondere", "oracle_pondere")
COULEURS = {"manuel": "#4a3aa7", "aleatoire": "#eb6834", "B2": "#2a78d6", "oracle": "#1baf7a",
            "B2_pondere": "#2a78d6", "oracle_pondere": "#1baf7a"}
TRAITS = {"B2": "--", "oracle": "--"}                    # score géométrique : pointillés
MARQUEURS = {"manuel": "o", "aleatoire": "D", "B2": "s", "oracle": "^", "B2_pondere": "s",
             "oracle_pondere": "^"}
LABELS = {"manuel": "Manuel (semaine 1)", "aleatoire": "Aléatoire",
          "B2": "B2, score géométrique", "oracle": "Oracle, score géométrique",
          "B2_pondere": "B2, score pondéré", "oracle_pondere": "Oracle, score pondéré"}
INK, INK_2 = "#0b0b0b", "#52514e"


# =============================================================================
# Étape 1 — Positions possibles et scores
# =============================================================================
def positions_possibles(cfg, te):
    """Positions du bloc test (pas de 2 m) où l'on peut poser un mât ou un support de toit."""
    g = cfg["grid"]
    x = g["xmin"] + (te["cols"] + 0.5) * g["res"]
    y = g["ymax"] - (te["rows"] + 0.5) * g["res"]
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs")
    bat = assets[assets["type"] == "batiment"]
    cm, ct = cfg["candidates"]["mat"], cfg["candidates"]["toit"]
    toit_raster = sample(ROOT / cfg["rasters"]["out_dir"] / "bati_mask.tif", np.c_[x, y]) == 1

    mat = (shapely.contains_xy(per.buffer(-cm["inset_m"]), x, y)
           & ~shapely.contains_xy(bat.union_all().buffer(cm["min_dist_building_m"]), x, y)
           & (te["strate"] != 0))
    nom_bat = np.full(x.size, None, object)
    for _, b in bat.iterrows():
        m = shapely.contains_xy(b.geometry.buffer(-ct["inset_m"]), x, y) & toit_raster
        nom_bat[m] = b["nom"]
    toit = (nom_bat != None) & (te["strate"] == 0)                       # noqa: E711
    return x, y, mat, toit, nom_bat


def scores(V, hauteurs, cfg, mat, toit):
    """max sur les hauteurs possibles du support de V̄_h (V : (n, h, anneau, secteur))."""
    Vb = potentiel_total(V, surfaces_bins(cfg))                          # (n, h)
    h = np.asarray(hauteurs, float)
    hm = np.isin(h, cfg["candidates"]["mat"]["heights_m"])
    ht = np.isin(h, cfg["candidates"]["toit"]["heights_m"])
    return np.where(mat, Vb[:, hm].max(1), np.where(toit, Vb[:, ht].max(1), -np.inf))


def masse_ponderee(cfg, rows, cols, lot=100):
    """M (n, anneau·secteur) : somme des poids w des cellules de chaque bin du disque de o."""
    from src.dataset_surrogate import bins_disque
    sc = cfg["surrogate"]
    dr, dc, b, n_bin = bins_disque(sc["rayon_m"], sc["anneaux_m"], sc["n_secteurs"],
                                   cfg["site"]["resolution_m"])
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    g = cfg["grid"]
    Wmap = np.zeros((g["height"], g["width"]))
    Wmap[cel["rows"], cel["cols"]] = cel["w"]
    onehot = sparse.csr_matrix((np.ones(b.size), (np.arange(b.size), b)), shape=(b.size, n_bin.size))
    M = np.zeros((rows.size, n_bin.size))
    for a in range(0, rows.size, lot):
        G = Wmap[rows[a:a + lot, None] + dr[None], cols[a:a + lot, None] + dc[None]]
        M[a:a + lot] = G @ onehot                        # dense × creux -> dense
    return M


def scores_ponderes(V, hauteurs, cfg, mat, toit, M):
    """Score pondéré (docstring du module). V : (n, h, anneau, secteur), M : (n, anneau·secteur)."""
    n, nh, na, ns = V.shape
    Mv = V * M.reshape(n, 1, na, ns)
    R = Mv.sum(axis=(2, 3))                                  # (n, h)
    C = Mv.sum(axis=2).max(axis=2)                           # (n, h)
    ok = mat | toit
    S = np.maximum(R / R[ok].max(), C / C[ok].max())
    h = np.asarray(hauteurs, float)
    hm = np.isin(h, cfg["candidates"]["mat"]["heights_m"])
    ht = np.isin(h, cfg["candidates"]["toit"]["heights_m"])
    return np.where(mat, S[:, hm].max(1), np.where(toit, S[:, ht].max(1), -np.inf))


# =============================================================================
# Étape 2 — Catalogues
# =============================================================================
def effectifs_manuels(cfg):
    c = gpd.read_file(ROOT / cfg["candidates"]["output"])
    return int((c["support"] != "toit").sum()), int((c["support"] == "toit").sum()), c


def selection(score, x, y, mat, toit, n_mat, n_toit, rayon, rng=None):
    """Indices retenus : suppression des non-maxima par type (ou tirage aléatoire si rng)."""
    out = []
    for m, k in ((mat, n_mat), (toit, n_toit)):
        idx = np.flatnonzero(m)
        if rng is not None:
            sc = rng.random(idx.size)                    # ordre aléatoire, même espacement minimal
        else:
            sc = score[idx]
        sel = nms(sc, x[idx], y[idx], k, rayon)
        assert sel.size == k, f"seulement {sel.size} positions espacées de {rayon} m (voulu : {k})"
        out.append(idx[sel])
    return out


def vers_candidats(cfg, x, y, i_mat, i_toit, nom_bat, prefixe):
    """Même schéma que data/site/candidats.gpkg (src.candidates.build_candidates)."""
    cm, ct = cfg["candidates"]["mat"], cfg["candidates"]["toit"]
    d = ROOT / cfg["rasters"]["out_dir"]
    idx = np.r_[i_mat, i_toit]
    is_roof = np.r_[np.zeros(i_mat.size, bool), np.ones(i_toit.size, bool)]
    xy = np.c_[x[idx], y[idx]]
    gdf = gpd.GeoDataFrame({"support": np.where(is_roof, "toit", "mat_interieur"),
                            "batiment": np.where(is_roof, nom_bat[idx], None)},
                           geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=cfg["lidar"]["crs"])
    gdf["z_sol"] = sample(d / "mnt.tif", xy)
    gdf["z_mns"] = sample(d / "mns.tif", xy)
    gdf["h_vegetation"] = np.where(is_roof, np.nan, sample(d / "autre_h.tif", xy))
    gdf["z_support"] = np.where(is_roof, gdf["z_mns"], gdf["z_sol"])
    gdf["hauteurs_m"] = np.where(is_roof, ",".join(map(str, ct["heights_m"])),
                                 ",".join(map(str, cm["heights_m"])))
    gdf["cout_base"] = np.where(is_roof, ct["cost_base"], cm["cost_base"])
    gdf["cout_par_m"] = np.where(is_roof, ct["cost_per_m"], cm["cost_per_m"])
    gdf.insert(0, "id", [f"{prefixe}{i:03d}" for i in range(len(gdf))])
    return gdf


# =============================================================================
# Étape 3 — Évaluation exacte d'un catalogue : matrice, glouton, pire chemin, NSGA-II
# =============================================================================
class Contexte:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = prepare_surfaces(cfg)
        d = ROOT / cfg["intrus"]["out_dir"]
        dm, gr = np.load(d / "domaine.npz"), np.load(d / "graphe.npz")
        self.dom = Domaine(rows=dm["rows"], cols=dm["cols"], interieur=dm["interieur"],
                           source=dm["source"], cible=dm["cible"], u=gr["u"], v=gr["v"],
                           longueur=gr["longueur"], franchit=gr["franchit"],
                           portail=gr["portail"])
        self.w_dom = dm["w"]
        self.n_int = int(self.dom.interieur.sum())
        assert self.dom.interieur[:self.n_int].all()
        cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
        self.w, self.zone = cells["w"], cells["zone"]
        b = cfg["optimisation"]["budget"]
        self.budgets = np.arange(b["min"], b["max"] + 1e-9, b["pas"])


def evaluer_catalogue(ctx: Contexte, cand, nom: str, out) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = ctx.cfg
    t0 = time.time()
    A, conf, _, _ = build_matrix(ctx.s, cand, ctx.dom.rows, ctx.dom.cols, ctx.w_dom, cfg)
    A = A.tocsr()
    t_A = time.time() - t0
    inst = from_configs(A[:, :ctx.n_int], ctx.w, conf, cfg)
    print(f"  {nom} : {len(cand)} positions, {A.shape[0]:,} configurations ({t_A:.0f} s)")

    rows, graines = [], []
    for B in ctx.budgets:
        sol = greedy(inst, B)
        assert not violations(inst, sol.T, B)
        P, _ = detection(A, sol.T)
        k = inst.kind[sol.T]
        rows.append(dict(catalogue=nom, budget=B, cout=sol.cost, F=sol.value / inst.W,
                         p_star_F=seuil_couloir(ctx.dom, P, "F")[0] if sol.T.size else 0.0,
                         p_star_G=seuil_couloir(ctx.dom, P, "G")[0] if sol.T.size else 0.0,
                         n_radar=int((k == KINDS.index("radar")).sum()),
                         n_camera=int((k == KINDS.index("camera")).sum()),
                         configs=" ".join(map(str, sol.T))))
        graines.append(sol.T)
    gl = pd.DataFrame(rows)
    front = nsga(ctx, A, conf, inst, graines, nom)
    return gl, front


def nsga(ctx: Contexte, A, conf, inst, graines, nom) -> pd.DataFrame:
    """NSGA-II de src.nsga, sur ce catalogue, amorcé par ses propres solutions gloutonnes."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    from src.nsga import (Amorcage, CroisementUnion, Evaluateur, MutationVoisinage, Placement,
                          RepairPlacement)

    cfg, dc = ctx.cfg, ctx.cfg["surrogate"]["decision"]
    ev = Evaluateur.__new__(Evaluateur)                 # même évaluateur, autre catalogue
    ev.cfg, ev.dom, ev.A, ev.conf = cfg, ctx.dom, A, conf
    ev.n_int, ev.inst, ev.zone = ctx.n_int, inst, ctx.zone
    k = cfg["nsga"]["k_couverture"]
    ev.k, ev.k_p, ev.k_part, ev.k_zones = k["k"], k["p_min"], k["part_min"], [int(z) for z in k["zones"]]
    ev.B_max = float(ctx.budgets[-1])

    fronts = []
    for g in dc["graines"]:
        rng = np.random.default_rng(g)
        algo = NSGA2(pop_size=int(dc["population"]), sampling=Amorcage(graines, rng),
                     crossover=CroisementUnion(rng, cfg["nsga"]["p_croisement"]),
                     mutation=MutationVoisinage(ev, rng), repair=RepairPlacement(ev, rng),
                     eliminate_duplicates=True)
        t0 = time.time()
        res = minimize(Placement(ev), algo, ("n_gen", int(dc["generations"])), seed=g,
                       verbose=False)
        for x in np.atleast_2d(res.X):
            T = np.flatnonzero(x.astype(bool))
            r = ev.evaluer(T)
            P, _ = detection(A, T)
            r["p_star_G"] = seuil_couloir(ctx.dom, P, "G")[0]
            fronts.append(dict(catalogue=nom, graine=g, **r, configs=" ".join(map(str, T))))
        print(f"    NSGA-II graine {g} : {time.time() - t0:.0f} s")
    f = pd.DataFrame(fronts).drop_duplicates("configs")
    from src.evaluation_intrus import non_dominated
    return f[non_dominated(f)].sort_values("cout").reset_index(drop=True)


# =============================================================================
# Étape 4 — Résumé et figures
# =============================================================================
def hypervolume(front, B_max):
    from pymoo.indicators.hv import HV
    Fo = np.c_[front.cout / B_max, -front.F, -front.p_star_F]
    return float(HV(ref_point=np.array([1.05, 0.0, 0.0]))(Fo))


def resume(gl: pd.DataFrame, fr: pd.DataFrame, B_max) -> pd.DataFrame:
    rows = []
    for nom, g in gl.groupby("catalogue", sort=False):
        f = fr[fr.catalogue == nom]
        r = dict(catalogue=nom, F_moyen_glouton=g.F.mean())
        for B in (13, 20, 30):
            gg = g[g.budget == B].iloc[0]
            r[f"F_glouton_B{B}"], r[f"pF_glouton_B{B}"] = gg.F, gg.p_star_F
        rob = f[f.p_star_F >= 0.9]
        r["seuil_robustesse_p0.9"] = rob.cout.min() if len(rob) else np.nan
        for B in (13, 20, 30):
            d = f[(f.cout <= B + 1e-9) & (f.p_star_F >= 0.95)]
            r[f"F_front_B{B}_p0.95"] = d.F.max() if len(d) else np.nan
        r["hypervolume"] = hypervolume(f, B_max)
        r["n_front"] = len(f)
        rows.append(r)
    return pd.DataFrame(rows)


def figures(gl, fr, cats, cfg, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation_intrus import style

    lab = LABELS
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    for ax, col, titre in zip(axs, ("F", "p_star_F"),
                              ("Couverture pondérée F (glouton)", "Détection garantie p*_F (glouton)")):
        style(ax)
        for nom, g in gl.groupby("catalogue", sort=False):
            ax.plot(g.budget, 100 * g[col], color=COULEURS[nom], lw=2, marker=MARQUEURS[nom],
                    ms=4, ls=TRAITS.get(nom, "-"), label=lab[nom],
                    mfc="white" if nom in TRAITS else COULEURS[nom])
        ax.set_xlabel("Budget (unités de coût)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title(titre, loc="left", fontsize=11, color=INK)
    axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_fig / "decision_budget.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    style(ax)
    for nom, f in fr.groupby("catalogue", sort=False):
        r = f[f.p_star_F >= 0.95].sort_values("cout")
        best = r.F.cummax()
        ax.step(r.cout, 100 * best, where="post", color=COULEURS[nom], lw=2,
                ls=TRAITS.get(nom, "-"), label=lab[nom])
        ax.scatter(r.cout, 100 * r.F, color=COULEURS[nom], marker=MARQUEURS[nom], s=14, zorder=3)
    ax.set_xlabel("Coût (unités)")
    ax.set_ylabel("Meilleure couverture F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Front NSGA-II restreint aux dispositifs robustes (p*_F ≥ 0,95)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_fig / "decision_front.png")
    plt.close(fig)

    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs")
    nc = 3
    nr = int(np.ceil(len(cats) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), dpi=150, squeeze=False)
    for ax in axs.ravel()[len(cats):]:
        ax.axis("off")
    for ax, (nom, (c, fond)) in zip(axs.ravel(), cats.items()):
        if fond is not None:
            x, y, v = fond
            sc = ax.scatter(x, y, c=100 * v, s=2.2, marker="s", cmap="Blues", vmin=0,
                            vmax=np.nanpercentile(100 * v, 99), lw=0)
        assets[assets["type"] == "batiment"].plot(ax=ax, color="#d9d8d3", lw=0)
        ax.plot(*per.exterior.xy, color=INK, lw=1)
        for sup, mk in (("toit", "^"), ("mat", "o")):
            m = c["support"].str.startswith(sup).to_numpy()
            ax.scatter(c.geometry.x[m], c.geometry.y[m], marker=mk, s=18, color=COULEURS[nom],
                       edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(lab[nom], loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Catalogues (ronds : mâts, triangles : toits) ; fond : score utilisé pour la "
                 "sélection", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_fig / "decision_carte.png")
    plt.close(fig)


# =============================================================================
# Étape 5 — Orchestration
# =============================================================================
def charger_b2(path):
    """Le modèle B2 est un pickle scikit-learn : il ne se recharge qu'avec la version qui l'a écrit.
    En cas d'échec, on demande de le régénérer dans l'environnement courant."""
    import warnings

    import sklearn
    import sklearn.exceptions
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", sklearn.exceptions.InconsistentVersionWarning)
            return joblib.load(path)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError,
            sklearn.exceptions.InconsistentVersionWarning) as e:
        raise SystemExit(f"{path.name} absent ou écrit par une autre version de scikit-learn "
                         f"({type(e).__name__}) : lance d'abord `python -m src.baselines_surrogate` "
                         f"(scikit-learn {sklearn.__version__}).") from None


def main() -> None:
    import yaml

    cfg = load_config()
    cfg["grid"] = yaml.safe_load(open(ROOT / cfg["site"]["grid_yaml"]))
    sc, dc = cfg["surrogate"], cfg["surrogate"]["decision"]
    noms = sys.argv[1:] or list(CATALOGUES)
    out = ROOT / sc["out_dir"] / "decision"
    out.mkdir(parents=True, exist_ok=True)
    out_fig = ROOT / cfg["optimisation"]["figures_dir"]
    out_fig.mkdir(parents=True, exist_ok=True)

    te = np.load(ROOT / sc["out_dir"] / "test.npz")
    x, y, mat, toit, nom_bat = positions_possibles(cfg, te)
    n_mat, n_toit, manuel = effectifs_manuels(cfg)
    print(f"Positions possibles (pas de 2 m) : {mat.sum():,} mâts, {toit.sum():,} toits ; "
          f"effectifs du catalogue manuel : {n_mat} mâts, {n_toit} toits")

    ctx = Contexte(cfg)
    hz = Horizons(ctx.s, cfg)
    pos = np.flatnonzero(mat | toit)
    t0 = time.time()
    V2 = np.zeros_like(te["Y"])
    V2[pos] = predire_b2(charger_b2(ROOT / sc["out_dir"] / "b2.joblib"), hz, te["rows"][pos],
                         te["cols"][pos], te["strate"][pos].astype(int), te["hauteurs"])
    print(f"Carte B2 sur {pos.size:,} positions : {time.time() - t0:.1f} s")
    s_b2 = scores(V2, te["hauteurs"], cfg, mat, toit)
    s_ex = scores(te["Y"], te["hauteurs"], cfg, mat, toit)
    ok = np.isfinite(s_ex)
    from scipy.stats import spearmanr
    print(f"Accord B2 / exact sur les positions possibles : Spearman {spearmanr(s_b2[ok], s_ex[ok])[0]:.3f}")

    M = np.zeros((te["rows"].size, V2[0].size // len(te["hauteurs"])))
    M[pos] = masse_ponderee(cfg, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
    w_b2 = scores_ponderes(V2, te["hauteurs"], cfg, mat, toit, M)
    w_ex = scores_ponderes(te["Y"], te["hauteurs"], cfg, mat, toit, M)
    print(f"Score pondéré : accord B2 / exact Spearman {spearmanr(w_b2[ok], w_ex[ok])[0]:.3f} ; "
          f"accord score pondéré / géométrique (exact) {spearmanr(w_ex[ok], s_ex[ok])[0]:.3f}")

    scores_cat = {"B2": s_b2, "oracle": s_ex, "B2_pondere": w_b2, "oracle_pondere": w_ex,
                  "aleatoire": None}
    cats = {}
    for nom in CATALOGUES:
        if nom not in noms:
            f = out / f"candidats_{nom}.gpkg"
            if nom == "manuel":
                cats[nom] = (manuel, None)
            elif f.exists() and (out / f"glouton_{nom}.csv").exists():
                cats[nom] = (gpd.read_file(f), None)          # évalué lors d'une exécution passée
            continue
        if nom == "manuel":
            cats[nom] = (manuel, None)
            continue
        rng = np.random.default_rng(dc["graine_aleatoire"]) if nom == "aleatoire" else None
        i_m, i_t = selection(scores_cat[nom], x, y, mat, toit, n_mat, n_toit, dc["nms_m"], rng)
        c = vers_candidats(cfg, x, y, i_m, i_t, nom_bat, prefixe=nom[0].upper())
        c.to_file(out / f"candidats_{nom}.gpkg", layer="candidats")
        sel = np.r_[i_m, i_t]
        print(f"  {nom:15s} : V̄ exact moyen {s_ex[sel].mean():.3f} ; score pondéré exact moyen "
              f"{w_ex[sel].mean():.3f}")
        fond = scores_cat[nom] if scores_cat[nom] is not None else w_ex
        cats[nom] = (c, (x[pos], y[pos], fond[pos]))

    gls, frs = [], []
    for nom, (c, fond) in cats.items():
        if nom in noms:
            gl, fr = evaluer_catalogue(ctx, c, nom, out)
            gl.to_csv(out / f"glouton_{nom}.csv", index=False)
            fr.to_csv(out / f"front_{nom}.csv", index=False)
        else:
            gl = pd.read_csv(out / f"glouton_{nom}.csv")
            fr = pd.read_csv(out / f"front_{nom}.csv")
        gls.append(gl)
        frs.append(fr)
    gl, fr = pd.concat(gls), pd.concat(frs)
    r = resume(gl, fr, float(ctx.budgets[-1]))
    r.to_csv(out / "resume.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + r.round(3).to_string(index=False))
    figures(gl, fr, cats, cfg, out_fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
