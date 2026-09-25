"""
src/eval_surrogate.py — Métriques communes aux références et au réseau (§8.5).

  précision : erreur absolue moyenne (EAM) sur V, par hauteur, strate et anneau
  classement : Spearman sur le potentiel total V̄_h (moyenne des bins pondérée par leur surface)
  décision   : top-K des positions à l'intérieur du périmètre, après suppression des non-maxima
               - rappel : part des K meilleures positions exactes retrouvées à moins de `tol_m`
               - regret : 1 − (V̄ exact des K choisies par le modèle) / (V̄ exact des K meilleures)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.dataset_surrogate import STRATES, bins_disque


def surfaces_bins(cfg) -> np.ndarray:
    sc = cfg["surrogate"]
    *_, n_bin = bins_disque(sc["rayon_m"], sc["anneaux_m"], sc["n_secteurs"],
                            cfg["site"]["resolution_m"])
    return n_bin.reshape(len(sc["anneaux_m"]) - 1, sc["n_secteurs"])


def potentiel_total(Y: np.ndarray, n_bin: np.ndarray) -> np.ndarray:
    """(n, h, anneau, secteur) -> (n, h) : part de tout le disque qui est du sol visible."""
    return np.einsum("nhas,as->nh", Y, n_bin) / n_bin.sum()


def nms(score, x, y, k: int, rayon: float) -> np.ndarray:
    """Indices des k meilleurs scores, deux à deux distants d'au moins `rayon`."""
    order = np.argsort(-score)
    pris = []
    for i in order:
        if all((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= rayon ** 2 for j in pris):
            pris.append(i)
            if len(pris) == k:
                break
    return np.asarray(pris)


def decision(Vp, Vt, x, y, k=20, rayon=10.0, tol=5.0) -> tuple[float, float]:
    ref = nms(Vt, x, y, k, rayon)
    sel = nms(Vp, x, y, k, rayon)
    d2 = (x[ref][:, None] - x[sel][None]) ** 2 + (y[ref][:, None] - y[sel][None]) ** 2
    rappel = float(np.mean(d2.min(axis=1) <= tol ** 2))
    regret = float(1.0 - Vt[sel].sum() / Vt[ref].sum())
    return rappel, regret


def rapport(Yp, Yt, strate, hauteurs, n_bin, interieur=None, xy=None, nom="") -> pd.DataFrame:
    """Une ligne par hauteur : EAM (globale, par anneau, par strate), Spearman sur V̄, décision."""
    Vp, Vt = potentiel_total(Yp, n_bin), potentiel_total(Yt, n_bin)
    err = np.abs(Yp - Yt)
    rows = []
    for ih, h in enumerate(hauteurs):
        r = dict(modele=nom, h=h, EAM=err[:, ih].mean(),
                 EAM_0_50=err[:, ih, 0].mean(), EAM_50_150=err[:, ih, 1].mean())
        for k, s in enumerate(STRATES):
            m = strate == k
            r[f"EAM_{s}"] = err[m, ih].mean() if m.any() else np.nan
        r["spearman_V"] = spearmanr(Vp[:, ih], Vt[:, ih])[0]
        if interieur is not None:
            m = interieur
            r["rappel_top20"], r["regret_top20"] = decision(Vp[m, ih], Vt[m, ih],
                                                            xy[0][m], xy[1][m])
        rows.append(r)
    return pd.DataFrame(rows)


def interieur_site(cfg, root, rows, cols):
    """Positions du bloc test à l'intérieur du périmètre, et leurs coordonnées."""
    import geopandas as gpd
    import shapely
    import yaml

    g = yaml.safe_load(open(root / cfg["site"]["grid_yaml"]))
    x = g["xmin"] + (cols + 0.5) * g["res"]
    y = g["ymax"] - (rows + 0.5) * g["res"]
    per = gpd.read_file(root / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    return shapely.contains_xy(per, x, y), (x, y)
