"""
src/baselines_surrogate.py — Références B0 et B1 du modèle de substitution (§8.5, étape 5.3).

B0 : moyenne des étiquettes d'entraînement par strate (toit, sol dégagé, sol sous végétation),
     pour chaque (hauteur, anneau, secteur).
B2 : le même boosting, sur les horizons de la pyramide log-polaire (src/horizons.py), calculés à
     chaque hauteur : l'écart B1 − B2 isole le coût de l'approximation par blocs.
B1 : gradient boosting sur des descripteurs locaux faits à la main. Pour chaque secteur de 45°
     et chaque bande de distance (0-10, 10-25, 25-50, 50-100, 100-150 m) :
       - tangente de l'angle d'horizon opaque   max (z_opaque − z_obs) / d
       - tangente de l'angle d'horizon végétal  max (z_veg − z_obs) / d
       - épaisseur optique verticale moyenne    moyenne de k · (z_veg − z_opaque)
       - part de sol
     Une ligne = (observateur, hauteur, secteur, anneau cible) ; le secteur n'est pas une
     variable : le modèle est invariant par rotation par construction.

Sorties (surrogate.out_dir) : baselines_test.npz (prédictions), baselines.csv (métriques),
                              b2.joblib (modèle B2, réutilisé par src.catalogue_surrogate)
Usage : python -m src.baselines_surrogate
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numba import njit, prange
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from src.dataset_surrogate import bins_disque
from src.eval_surrogate import interieur_site, rapport, surfaces_bins
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces

BANDES_M = (0.0, 10.0, 25.0, 50.0, 100.0, 150.0)
N_DESC = 4


# =============================================================================
# Étape 1 — Descripteurs par (observateur, hauteur, secteur)
# =============================================================================
def offsets_bandes(cfg):
    sc, res = cfg["surrogate"], cfg["site"]["resolution_m"]
    dr, dc, b, _ = bins_disque(sc["rayon_m"], sc["anneaux_m"], sc["n_secteurs"], res)
    sect = b % sc["n_secteurs"]
    d = np.hypot(dr, dc) * res
    bande = np.searchsorted(np.asarray(BANDES_M[1:-1]), d, side="left")
    return dr, dc, sect, bande, d


@njit(parallel=True, cache=True)
def _descripteurs(z_opaque, z_veg, k_ext, sol, obs_r, obs_c, hauteurs, dr, dc, sect, bande, d,
                  n_sect, n_bande):
    n, nh = obs_r.size, hauteurs.size
    out = np.zeros((n, nh, n_sect, n_bande * 4), np.float32)
    for i in prange(n):
        r0, c0 = obs_r[i], obs_c[i]
        t_o = np.full((nh, n_sect, n_bande), -1e9)
        t_v = np.full((nh, n_sect, n_bande), -1e9)
        s_k = np.zeros((n_sect, n_bande))
        s_s = np.zeros((n_sect, n_bande))
        cnt = np.zeros((n_sect, n_bande))
        for m in range(dr.size):
            r, c = r0 + dr[m], c0 + dc[m]
            s, b = sect[m], bande[m]
            zo, zv = z_opaque[r, c], z_veg[r, c]
            for ih in range(nh):
                z = z_opaque[r0, c0] + hauteurs[ih]
                a = (zo - z) / d[m]
                if a > t_o[ih, s, b]:
                    t_o[ih, s, b] = a
                a = (zv - z) / d[m]
                if a > t_v[ih, s, b]:
                    t_v[ih, s, b] = a
            s_k[s, b] += k_ext[r, c] * (zv - zo)
            s_s[s, b] += 1.0 if sol[r, c] else 0.0
            cnt[s, b] += 1.0
        for ih in range(nh):
            for s in range(n_sect):
                for b in range(n_bande):
                    out[i, ih, s, 4 * b] = t_o[ih, s, b]
                    out[i, ih, s, 4 * b + 1] = t_v[ih, s, b]
                    out[i, ih, s, 4 * b + 2] = s_k[s, b] / cnt[s, b]
                    out[i, ih, s, 4 * b + 3] = s_s[s, b] / cnt[s, b]
    return out


def descripteurs(s, cfg, rows, cols, lot=5000):
    dr, dc, sect, bande, d = offsets_bandes(cfg)
    sol = s.z_opaque < s.z_tgt
    h = np.asarray(cfg["surrogate"]["hauteurs_m"], np.float64)
    ns, nb = cfg["surrogate"]["n_secteurs"], len(BANDES_M) - 1
    out = np.empty((rows.size, h.size, ns, nb * N_DESC), np.float32)
    for a in range(0, rows.size, lot):
        sl = slice(a, a + lot)
        out[sl] = _descripteurs(s.z_opaque, s.z_veg, s.k_ext, sol, rows[sl].astype(np.int64),
                                cols[sl].astype(np.int64), h, dr, dc, sect, bande, d, ns, nb)
    return out


def table(D, strate, hauteurs, n_anneaux):
    """(n, h, s, f) -> lignes (n·h·s·anneau, f + 3) : descripteurs, strate, h, anneau cible."""
    n, nh, ns, nf = D.shape
    X = np.repeat(D.reshape(n * nh * ns, nf), n_anneaux, axis=0)
    st = np.repeat(strate, nh * ns * n_anneaux)
    hh = np.tile(np.repeat(hauteurs, ns * n_anneaux), n)
    an = np.tile(np.arange(n_anneaux), n * nh * ns)
    return np.column_stack([X, st, hh, an]).astype(np.float32)


def cible(Y):
    """(n, h, anneau, secteur) -> (n·h·s·anneau,), même ordre que `table`."""
    return np.transpose(Y, (0, 1, 3, 2)).reshape(-1)


def depuis_table(p, n, nh, na, ns):
    return np.transpose(p.reshape(n, nh, ns, na), (0, 1, 3, 2))


def descripteurs_pyramide(hz, rows, cols):
    """Descripteurs de B2 : (n, h, s, 2·b horizons + 2·b moyennes), même disposition que `table`."""
    hor, opa = hz(rows, cols)                                  # (n, h, 2, b, s), (n, 2, b, s)
    n, nh, ns = len(rows), hz.h.size, hz.ns
    hor = hor.transpose(0, 1, 4, 3, 2).reshape(n, nh, ns, -1)
    opa = np.repeat(opa.transpose(0, 3, 2, 1).reshape(n, ns, -1)[:, None], nh, axis=1)
    return np.concatenate([hor, opa], axis=-1)


def predire_b2(modele, hz, rows, cols, strate, hauteurs, n_anneaux=2):
    """V prédit par B2 : (n, h, anneau, secteur)."""
    D = descripteurs_pyramide(hz, rows, cols)
    p = modele.predict(table(D, strate, hauteurs, n_anneaux))
    return np.clip(depuis_table(p, len(rows), len(hauteurs), n_anneaux, hz.ns), 0, 1)


# =============================================================================
# Étape 2 — Entraînement et évaluation
# =============================================================================
def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["surrogate"]["out_dir"]
    tr = np.load(out / "etiquettes.npz")
    te = np.load(out / "test.npz")
    Y, st, sp = tr["Y"], tr["strate"].astype(int), tr["split"]
    h = tr["hauteurs"]
    _, nh, na, ns = Y.shape
    n_bin = surfaces_bins(cfg)
    Yt, stt = te["Y"], te["strate"].astype(int)
    inter, xy = interieur_site(cfg, ROOT, te["rows"], te["cols"])
    ent, val = sp == 0, sp == 1

    # --- B0
    moy = np.stack([Y[ent & (st == k)].mean(axis=0) for k in range(3)])
    P0 = moy[stt]
    P0v = moy[st[val]]

    # --- B1
    s = prepare_surfaces(cfg)
    t0 = time.time()
    D = descripteurs(s, cfg, tr["rows"], tr["cols"])
    t_desc = time.time() - t0
    Dt = descripteurs(s, cfg, te["rows"], te["cols"])
    X, y = table(D[ent], st[ent], h, na), cible(Y[ent])
    print(f"B1 : {X.shape[0]:,} lignes × {X.shape[1]} variables ; descripteurs "
          f"{1e3 * t_desc / len(Y):.2f} ms par position")
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                       categorical_features=[X.shape[1] - 3],
                                       random_state=cfg["project"]["random_seed"])
    t0 = time.time()
    gb.fit(X, y)
    t_fit = time.time() - t0
    t0 = time.time()
    P1 = np.clip(depuis_table(gb.predict(table(Dt, stt, h, na)), len(Yt), nh, na, ns), 0, 1)
    t_pred = time.time() - t0
    P1v = np.clip(depuis_table(gb.predict(table(D[val], st[val], h, na)), val.sum(), nh, na, ns),
                  0, 1)
    print(f"  entraînement {t_fit:.0f} s ; prédiction {1e3 * t_pred / len(Yt):.2f} ms par position "
          f"(+ descripteurs)")

    # --- B2 : mêmes lignes, descripteurs de la pyramide : horizons à chaque h + moyennes
    from src.horizons import Horizons
    hz = Horizons(s, cfg)
    desc_pyr = lambda rows, cols: descripteurs_pyramide(hz, rows, cols)
    Dp = desc_pyr(tr["rows"], tr["cols"])
    t0 = time.time()
    Dpt = desc_pyr(te["rows"], te["cols"])
    t_desc2 = (time.time() - t0) / len(Yt)
    gb2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                        categorical_features=[Dp.shape[-1]],
                                        random_state=cfg["project"]["random_seed"])
    gb2.fit(table(Dp[ent], st[ent], h, na), y)
    joblib.dump(gb2, out / "b2.joblib")                          # réutilisé en 6.3
    t0 = time.time()
    P2 = np.clip(depuis_table(gb2.predict(table(Dpt, stt, h, na)), len(Yt), nh, na, ns), 0, 1)
    t_pred2 = (time.time() - t0) / len(Yt)
    print(f"B2 : descripteurs {1e3 * t_desc2:.3f} ms par position (positions du test, non contiguës), "
          f"prédiction {1e3 * t_pred2:.3f} ms")
    P2v = np.clip(depuis_table(gb2.predict(table(Dp[val], st[val], h, na)), val.sum(), nh, na, ns),
                  0, 1)

    res = pd.concat([
        rapport(P0v, Y[val], st[val], h, n_bin, nom="B0 (validation)"),
        rapport(P1v, Y[val], st[val], h, n_bin, nom="B1 (validation)"),
        rapport(P0, Yt, stt, h, n_bin, inter, xy, nom="B0 (test)"),
        rapport(P1, Yt, stt, h, n_bin, inter, xy, nom="B1 (test)"),
        rapport(P2v, Y[val], st[val], h, n_bin, nom="B2 (validation)"),
        rapport(P2, Yt, stt, h, n_bin, inter, xy, nom="B2 (test)"),
    ], ignore_index=True)
    res["cout_ms_position"] = np.where(res.modele.str.startswith("B1"),
                                       1e3 * (t_desc / len(Y) + t_pred / len(Yt)), 0.0)
    # B2 : descripteurs de la pyramide + prédiction du boosting (dominée par le boosting)
    res.loc[res.modele.str.startswith("B2"), "cout_ms_position"] = 1e3 * (t_desc2 + t_pred2)
    res.to_csv(out / "baselines.csv", index=False)
    np.savez_compressed(out / "baselines_test.npz", B0=P0.astype(np.float32),
                        B1=P1.astype(np.float32), B2=P2.astype(np.float32))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
