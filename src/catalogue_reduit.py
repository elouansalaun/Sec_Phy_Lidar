"""
src/catalogue_reduit.py — Catalogues réduits : la sélection compte-t-elle quand les positions
candidates sont rares ? (§8.9, étape 6.3c)

Avec 61 positions, un site de 1,5 ha est saturé : un catalogue tiré au hasard fait aussi bien que le
catalogue manuel et que B2 (6.3b). Hypothèse : l'avantage d'une sélection informée n'apparaît que
lorsque le nombre de positions est petit. On compare, pour K positions (même proportion mâts / toits
que le catalogue manuel) :
  B2_pondere, oracle_pondere : suppression des non-maxima sur le score pondéré (src.catalogue_surrogate)
  aleatoire × graines        : positions possibles tirées au hasard, même espacement minimal
  manuel_sous × graines      : K positions tirées dans le catalogue manuel de la semaine 1
Évaluation exacte et identique : matrice sur le domaine de l'intrus, glouton sur la grille de budgets,
p*_F et p*_G de chaque solution (pas de NSGA-II).

Mode `protocole: "preenregistre"` (6.3d, décidé AVANT exécution, config surrogate.decision.reduit) :
  espacement   R_type = facteur · √(A_type / K_type), A_type = surface des positions possibles du type
               (mêmes R pour les catalogues sélectionnés ET pour le témoin aléatoire)
  bande        cellules extérieures du domaine de l'intrus ajoutées à la carte des poids, masse totale
               = masse_bande × masse intérieure, répartie uniformément
  catalogues   B2_bande (candidat), oracle_bande, B2_pondere (sans bande), aleatoire, manuel_sous
  succès       F moyen de B2_bande > max des graines aléatoires pour ≥ 3 des 4 tailles
Mode `protocole: "initial"` (6.3c) : espacement fixe `nms_m`, sans bande.

Sorties (surrogate.out_dir/decision) : reduit.csv (une ligne par catalogue et budget),
                                       resume_reduit.csv ; figure : figures/decision_reduit.png
  (mode préenregistré : suffixe _preenregistre sur les trois)
Usage : python -m src.catalogue_reduit
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.baselines_surrogate import predire_b2
from src.catalogue_surrogate import (Contexte, charger_b2, effectifs_manuels, masse_ponderee,
                                     positions_possibles, scores_ponderes, selection, vers_candidats)
from src.coverage import build_matrix
from src.greedy import greedy
from src.horizons import Horizons
from src.instance import from_configs, violations
from src.intrus import detection, seuil_couloir
from src.site import ROOT, load_config

COULEURS = {"B2_pondere": "#2a78d6", "oracle_pondere": "#1baf7a", "aleatoire": "#eb6834",
            "manuel_sous": "#4a3aa7", "B2_bande": "#2a78d6", "oracle_bande": "#1baf7a"}
TRAITS = {"B2_pondere": "--"}
LABELS = {"B2_pondere": "B2, score pondéré", "oracle_pondere": "Oracle, score pondéré",
          "aleatoire": "Aléatoire (5 graines)", "manuel_sous": "Sous-catalogue manuel (5 graines)",
          "B2_bande": "B2, score pondéré + bande", "oracle_bande": "Oracle, score pondéré + bande"}
INK, INK_2 = "#0b0b0b", "#52514e"


# =============================================================================
# Étape 0 — Protocole préenregistré : carte des poids avec bande, espacement par règle
# =============================================================================
def carte_poids(cfg, masse_bande: float) -> np.ndarray:
    """Poids intérieurs (cellules.npz) + bande extérieure du domaine de l'intrus, de masse totale
    masse_bande × masse intérieure répartie uniformément."""
    g = cfg["grid"]
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    dm = np.load(ROOT / cfg["intrus"]["out_dir"] / "domaine.npz")
    W = np.zeros((g["height"], g["width"]))
    W[cel["rows"], cel["cols"]] = cel["w"]
    ext = ~dm["interieur"]
    if masse_bande > 0:
        W[dm["rows"][ext], dm["cols"][ext]] = masse_bande * cel["w"].sum() / ext.sum()
    return W


def masse(cfg, W, rows, cols, lot=100):
    """M (n, anneau·secteur) : somme de W sur chaque bin du disque (cf. catalogue_surrogate)."""
    from scipy import sparse

    from src.dataset_surrogate import bins_disque
    sc = cfg["surrogate"]
    dr, dc, b, n_bin = bins_disque(sc["rayon_m"], sc["anneaux_m"], sc["n_secteurs"],
                                   cfg["site"]["resolution_m"])
    onehot = sparse.csr_matrix((np.ones(b.size), (np.arange(b.size), b)),
                               shape=(b.size, n_bin.size))
    M = np.zeros((rows.size, n_bin.size))
    for a in range(0, rows.size, lot):
        M[a:a + lot] = W[rows[a:a + lot, None] + dr[None], cols[a:a + lot, None] + dc[None]] @ onehot
    return M


def rayons(mat, toit, n_mat, n_toit, facteur, pas_m=2.0):
    """R = facteur · √(A / K) par type ; A = nombre de positions possibles × pas²."""
    return (facteur * np.sqrt(mat.sum() * pas_m ** 2 / max(n_mat, 1)),
            facteur * np.sqrt(toit.sum() * pas_m ** 2 / max(n_toit, 1)))


def selection2(score, x, y, mat, toit, n_mat, n_toit, R_mat, R_toit, rng=None):
    from src.eval_surrogate import nms
    out = []
    for m, k, R in ((mat, n_mat, R_mat), (toit, n_toit, R_toit)):
        idx = np.flatnonzero(m)
        sc = rng.random(idx.size) if rng is not None else score[idx]
        j = nms(sc, x[idx], y[idx], k, R)
        assert j.size == k, f"{j.size} positions espacées de {R:.1f} m (voulu : {k})"
        out.append(idx[j])
    return out


# =============================================================================
# Étape 1 — Évaluation gloutonne exacte d'un catalogue
# =============================================================================
def evaluer_glouton(ctx: Contexte, cand) -> pd.DataFrame:
    cfg = ctx.cfg
    A, conf, _, _ = build_matrix(ctx.s, cand, ctx.dom.rows, ctx.dom.cols, ctx.w_dom, cfg)
    A = A.tocsr()
    inst = from_configs(A[:, :ctx.n_int], ctx.w, conf, cfg)
    rows = []
    for B in ctx.budgets:
        sol = greedy(inst, B)
        assert not violations(inst, sol.T, B)
        P, _ = detection(A, sol.T)
        rows.append(dict(budget=B, cout=sol.cost, F=sol.value / inst.W,
                         p_star_F=seuil_couloir(ctx.dom, P, "F")[0] if sol.T.size else 0.0,
                         p_star_G=seuil_couloir(ctx.dom, P, "G")[0] if sol.T.size else 0.0))
    return pd.DataFrame(rows)


def resumer(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (K, catalogue, graine), puis moyenne / min / max sur les graines."""
    def une(g):
        rob = g[g.p_star_F >= 0.9]
        return pd.Series(dict(F_moyen=g.F.mean(),
                              F_B13=g.loc[g.budget == 13, "F"].iloc[0],
                              F_B20=g.loc[g.budget == 20, "F"].iloc[0],
                              pF_B13=g.loc[g.budget == 13, "p_star_F"].iloc[0],
                              pF_B20=g.loc[g.budget == 20, "p_star_F"].iloc[0],
                              budget_robuste=rob.budget.min() if len(rob) else np.nan))
    par = df.groupby(["K", "catalogue", "graine"]).apply(une, include_groups=False).reset_index()
    agg = par.groupby(["K", "catalogue"]).agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return par, agg.drop(columns=[c for c in agg.columns if c.startswith("graine")]).reset_index()


# =============================================================================
# Étape 2 — Figure
# =============================================================================
def figure(par: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation_intrus import style

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, col, titre, pct in zip(axs, ("F_moyen", "F_B13", "budget_robuste"),
                                   ("F moyen sur les budgets (glouton)", "F à B = 13",
                                    "Premier budget avec p*_F ≥ 0,9"), (True, True, False)):
        style(ax)
        Ks = sorted(par.K.unique())
        for j, (nom, g) in enumerate(par.groupby("catalogue", sort=False)):
            n_c = par.catalogue.nunique()
            dx = (j - (n_c - 1) / 2) * 0.6 / n_c
            m = g.groupby("K")[col].agg(["mean", "min", "max"]).reindex(Ks)
            k = 100 if pct else 1
            xs = np.arange(len(Ks)) + dx
            ax.plot(xs, k * m["mean"], color=COULEURS[nom], lw=2, marker="o", ms=5,
                    ls=TRAITS.get(nom, "-"), label=LABELS[nom])
            if (m["max"] > m["min"]).any():
                ax.vlines(xs, k * m["min"], k * m["max"], color=COULEURS[nom], lw=1.2, alpha=0.8)
        ax.set_xticks(range(len(Ks)), [f"K = {k}" for k in Ks])
        ax.set_title(titre, loc="left", fontsize=11, color=INK)
        if pct:
            ax.set_ylabel("%")
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("Catalogues réduits : moyenne et étendue sur les graines (traits verticaux)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Étape 3 — Orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    cfg["grid"] = yaml.safe_load(open(ROOT / cfg["site"]["grid_yaml"]))
    sc, dc = cfg["surrogate"], cfg["surrogate"]["decision"]
    rc = dc["reduit"]
    out = ROOT / sc["out_dir"] / "decision"
    out.mkdir(parents=True, exist_ok=True)

    te = np.load(ROOT / sc["out_dir"] / "test.npz")
    x, y, mat, toit, nom_bat = positions_possibles(cfg, te)
    n_mat61, n_toit61, manuel = effectifs_manuels(cfg)
    ctx = Contexte(cfg)
    pos = np.flatnonzero(mat | toit)
    V2 = np.zeros_like(te["Y"])
    V2[pos] = predire_b2(charger_b2(ROOT / sc["out_dir"] / "b2.joblib"), Horizons(ctx.s, cfg),
                         te["rows"][pos], te["cols"][pos], te["strate"][pos].astype(int),
                         te["hauteurs"])
    M = np.zeros((te["rows"].size, V2[0].size // len(te["hauteurs"])))
    M[pos] = masse_ponderee(cfg, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
    w_b2 = scores_ponderes(V2, te["hauteurs"], cfg, mat, toit, M)
    w_ex = scores_ponderes(te["Y"], te["hauteurs"], cfg, mat, toit, M)
    ok = np.isfinite(w_ex)
    print(f"Score pondéré : Spearman B2 / exact = {spearmanr(w_b2[ok], w_ex[ok])[0]:.3f}")

    m_man = (manuel["support"] != "toit").to_numpy()
    prereg = rc.get("protocole", "initial") == "preenregistre"
    if prereg:
        W = carte_poids(cfg, rc["masse_bande"])
        Mb = np.zeros_like(M)
        Mb[pos] = masse(cfg, W, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
        b_b2 = scores_ponderes(V2, te["hauteurs"], cfg, mat, toit, Mb)
        b_ex = scores_ponderes(te["Y"], te["hauteurs"], cfg, mat, toit, Mb)
        print(f"Score pondéré + bande : Spearman B2 / exact = {spearmanr(b_b2[ok], b_ex[ok])[0]:.3f} ; "
              f"avec / sans bande (exact) = {spearmanr(b_ex[ok], w_ex[ok])[0]:.3f}")
        choisis = (("B2_bande", b_b2), ("oracle_bande", b_ex), ("B2_pondere", w_b2))
    else:
        choisis = (("B2_pondere", w_b2), ("oracle_pondere", w_ex))

    rows, t0 = [], time.time()
    for K in (rc["tailles_preenregistre"] if prereg else rc["tailles"]):
        n_mat = int(round(K * n_mat61 / (n_mat61 + n_toit61)))
        n_toit = K - n_mat
        if prereg:
            Rm, Rt = rayons(mat, toit, n_mat, n_toit, rc["facteur_espacement"])
        else:
            Rm = Rt = dc["nms_m"]
        print(f"K = {K} : {n_mat} mâts (espacement {Rm:.1f} m), {n_toit} toits ({Rt:.1f} m)")
        cats = []
        for nom, sc_ in choisis:
            i_m, i_t = selection2(sc_, x, y, mat, toit, n_mat, n_toit, Rm, Rt)
            cats.append((nom, 0, vers_candidats(cfg, x, y, i_m, i_t, nom_bat, "R")))
        for g in rc["graines"]:
            rng = np.random.default_rng(g)
            i_m, i_t = selection2(None, x, y, mat, toit, n_mat, n_toit, Rm, Rt, rng)
            cats.append(("aleatoire", g, vers_candidats(cfg, x, y, i_m, i_t, nom_bat, "R")))
            idx = np.r_[rng.choice(np.flatnonzero(m_man), n_mat, replace=False),
                        rng.choice(np.flatnonzero(~m_man), n_toit, replace=False)]
            cats.append(("manuel_sous", g, manuel.iloc[idx].reset_index(drop=True)))
        for nom, g, c in cats:
            df = evaluer_glouton(ctx, c).assign(K=K, catalogue=nom, graine=g)
            rows.append(df)
            print(f"  K = {K:2d}  {nom:15s} graine {g}  F moyen {df.F.mean():.3f}   "
                  f"p*_F(B=20) {df.loc[df.budget == 20, 'p_star_F'].iloc[0]:.2f}"
                  f"   [{time.time() - t0:5.0f} s]", flush=True)
    df = pd.concat(rows, ignore_index=True)
    par, agg = resumer(df)
    if not prereg:                                  # 6.3d écrit ses propres fichiers (plus bas)
        df.to_csv(out / "reduit.csv", index=False)
        agg.to_csv(out / "resume_reduit.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["K", "catalogue"] + [c for c in agg.columns if c.endswith("_mean")]
    print("\n" + agg[cols].round(3).to_string(index=False))
    suffixe = "_preenregistre" if prereg else ""
    if prereg:
        # Critère de succès préenregistré
        ok_K = []
        for K, g in par.groupby("K"):
            b2 = g.loc[g.catalogue == "B2_bande", "F_moyen"].iloc[0]
            al = g.loc[g.catalogue == "aleatoire", "F_moyen"]
            ok_K.append(b2 > al.max())
            print(f"  K = {K:2d} : B2_bande {b2:.3f}  vs aléatoire max {al.max():.3f} "
                  f"(moyenne {al.mean():.3f}) -> {'bat' if b2 > al.max() else 'ne bat pas'}")
        verdict = sum(ok_K) >= 3
        print(f"CRITÈRE PRÉENREGISTRÉ (≥ 3 tailles sur 4) : {sum(ok_K)} / {len(ok_K)} -> "
              f"{'SUCCÈS' if verdict else 'ÉCHEC'}")
        agg.to_csv(out / "resume_reduit_preenregistre.csv", index=False)
        df.to_csv(out / "reduit_preenregistre.csv", index=False)
    figure(par, ROOT / cfg["optimisation"]["figures_dir"] / f"decision_reduit{suffixe}.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
