"""
src/horizons.py — Horizons denses par pyramide log-polaire (étape 6.1, §8.8).

Pour chaque cellule o, chaque hauteur h et chaque (bande de distance b, secteur s) :
  horizon opaque   A_o[h, b, s](o) = atan( max_{j ∈ coin(b,s)} (z_opaque(j) − z_opaque(o) − h) / d_oj )
  horizon végétal  A_v[h, b, s](o) = idem avec z_veg
  (le terme −h/d domine près de l'observateur : un horizon calculé à h = 0 ne suffit pas, §8.8)
  part de sol      S[b, s](o)     = moyenne du masque sol sur le coin (borne supérieure de V)
  opacité          O[b, s](o)   = log(1 + moyenne sur le coin de k·(z_veg − z_opaque))

Calcul exact : 70 680 cellules lues par position. Pyramide : dans la bande b, on lit un bloc carré
de p_b × p_b cellules (p_b impair) tous les p_b pixels, centré sur o + p_b·(i, j) : dans une bande,
les blocs pavent exactement le plan ; aux frontières entre bandes de pas différents, ils se
chevauchent ou laissent des trous (configuration par défaut : 92,7 % des cellules du disque
couvertes une fois, 4,6 % non couvertes, 2,7 % deux fois ; tests/test_horizons.py). Chaque bloc porte le max (ou la moyenne) de ses cellules, calculé
une fois pour toute la grille par un filtre glissant. La distance est celle du centre du bloc :
l'erreur n'a PAS de signe fixe (la cellule la plus haute d'un bloc peut être plus proche ou plus
loin que son centre) ; elle est mesurée par `python -m src.horizons` (≈ 1° en moyenne, bande
0-10 m exacte). Coût : ≈ 1 200 lectures par position au lieu de 70 680 ; en float32 et sur des
positions contiguës (cas d'une carte dense), ≈ 48 fois moins cher que l'horizon exact.

Le réseau voit ces cartes comme des canaux d'entrée ; l'observateur à la hauteur h, la courbure
exacte des rayons et la végétation restent à sa charge.

Usage : python -m src.horizons     (mesure du coût et de l'erreur contre l'horizon exact)
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange
from scipy.ndimage import maximum_filter, uniform_filter

N_FEAT = 3            # horizon opaque, horizon végétal, opacité


# =============================================================================
# Étape 1 — Réseau de blocs : pour chaque bande, blocs de côté p_b tous les p_b pixels
# =============================================================================
def reseau_blocs(bandes_m, pas_px, n_secteurs: int, res: float):
    """Offsets des centres de blocs, niveau (indice de p_b), bande, secteur, distance [m]."""
    dr, dc, lvl, band, sect, dist = [], [], [], [], [], []
    w = 360.0 / n_secteurs
    for b, (lo, hi) in enumerate(zip(bandes_m[:-1], bandes_m[1:])):
        p = int(pas_px[b])
        assert p % 2 == 1, "pas impair : le bloc doit être centré sur une cellule"
        k = int(np.ceil(hi / res / p)) + 1
        i, j = np.mgrid[-k:k + 1, -k:k + 1]
        r, c = (i * p).ravel(), (j * p).ravel()
        d = np.hypot(r, c) * res
        keep = (d > lo) & (d <= hi) if b > 0 else (d > 0) & (d <= hi)
        r, c, d = r[keep], c[keep], d[keep]
        az = np.degrees(np.arctan2(c, -r)) % 360.0
        dr.append(r); dc.append(c); dist.append(d)
        sect.append(np.floor((az + w / 2) / w).astype(np.int64) % n_secteurs)
        band.append(np.full(r.size, b)); lvl.append(np.full(r.size, b))
    cat = lambda a, t: np.concatenate(a).astype(t)
    return (cat(dr, np.int64), cat(dc, np.int64), cat(lvl, np.int64), cat(band, np.int64),
            cat(sect, np.int64), cat(dist, np.float64))


def pyramide(s, pas_px):
    """Filtres glissants par niveau : max de z_opaque, max de z_veg, moyennes de l'opacité et du sol.
    Les deux moyennes sont empilées : om[0] opacité, om[1] sol."""
    can = np.maximum(s.z_veg - s.z_opaque, 0.0)
    op = s.k_ext * can
    sol = (s.z_opaque < s.z_tgt).astype(np.float64)
    f = lambda fn, a: np.stack([fn(a, size=p, mode="nearest") for p in pas_px]).astype(np.float32)
    return (f(maximum_filter, s.z_opaque), f(maximum_filter, s.z_veg),
            np.stack([f(uniform_filter, op), f(uniform_filter, sol)]))


# =============================================================================
# Étape 2 — Noyau dense
# =============================================================================
@njit(parallel=True, cache=True)
def _horizons(z_opaque, zo, zv, om, rows, cols, hauteurs, dr, dc, lvl, band, sect, dist, n_b, n_s):
    """-> horizons (n, n_h, 2, n_b, n_s) et moyennes (n, 2, n_b, n_s) : opacité, sol. Hors grille,
    les blocs sont ignorés : près du bord, l'horizon est partiel (positions jamais évaluées)."""
    n, nh = rows.size, hauteurs.size
    H, W = z_opaque.shape
    hor = np.zeros((n, nh, 2, n_b, n_s), np.float32)
    opa = np.zeros((n, 2, n_b, n_s), np.float32)
    for i in prange(n):
        r0, c0 = rows[i], cols[i]
        z0 = z_opaque[r0, c0]
        ho = np.full((nh, n_b, n_s), -1e9)
        hv = np.full((nh, n_b, n_s), -1e9)
        so = np.zeros((n_b, n_s))
        ss = np.zeros((n_b, n_s))
        cnt = np.zeros((n_b, n_s))
        for m in range(dr.size):
            r, c = r0 + dr[m], c0 + dc[m]
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            L, b, s = lvl[m], band[m], sect[m]
            do, dv = zo[L, r, c] - z0, zv[L, r, c] - z0
            for ih in range(nh):
                a = (do - hauteurs[ih]) / dist[m]
                if a > ho[ih, b, s]:
                    ho[ih, b, s] = a
                a = (dv - hauteurs[ih]) / dist[m]
                if a > hv[ih, b, s]:
                    hv[ih, b, s] = a
            so[b, s] += om[0, L, r, c]
            ss[b, s] += om[1, L, r, c]
            cnt[b, s] += 1.0
        for b in range(n_b):
            for s in range(n_s):
                opa[i, 0, b, s] = np.log1p(so[b, s] / max(cnt[b, s], 1.0))
                opa[i, 1, b, s] = ss[b, s] / max(cnt[b, s], 1.0)
                for ih in range(nh):
                    hor[i, ih, 0, b, s] = np.arctan(ho[ih, b, s]) if cnt[b, s] > 0 else 0.0
                    hor[i, ih, 1, b, s] = np.arctan(hv[ih, b, s]) if cnt[b, s] > 0 else 0.0
    return hor, opa


class Horizons:
    """Pyramide préparée une fois ; appels répétés sur des listes de positions."""

    def __init__(self, s, cfg):
        hc, sc = cfg["surrogate"]["horizons"], cfg["surrogate"]
        self.blocs = reseau_blocs(hc["bandes_m"], hc["pas_px"], sc["n_secteurs"], s.grid.res)
        t0 = time.time()
        self.zo, self.zv, self.om = pyramide(s, hc["pas_px"])
        self.t_prep = time.time() - t0
        self.z32 = s.z_opaque.astype(np.float32)
        self.h = np.asarray(sc["hauteurs_m"], np.float32)
        self.nb, self.ns = len(hc["bandes_m"]) - 1, sc["n_secteurs"]
        self.forme = s.z_opaque.shape

    def __call__(self, rows, cols, lot=200_000):
        n = rows.size
        hor = np.empty((n, self.h.size, 2, self.nb, self.ns), np.float32)
        opa = np.empty((n, 2, self.nb, self.ns), np.float32)
        for a in range(0, n, lot):
            sl = slice(a, a + lot)
            hor[sl], opa[sl] = _horizons(self.z32, self.zo, self.zv, self.om,
                                         rows[sl].astype(np.int64), cols[sl].astype(np.int64),
                                         self.h, *self.blocs, self.nb, self.ns)
        return hor, opa

    def cartes(self, lignes: int = 64):
        """Grille complète, en float16, calculée par blocs de lignes (mémoire bornée) :
        horizons (h, 2, b, s, H, W), moyennes (2, b, s, H, W)."""
        H, W = self.forme
        hor = np.empty((self.h.size, 2, self.nb, self.ns, H, W), np.float16)
        moy = np.empty((2, self.nb, self.ns, H, W), np.float16)
        t0 = time.time()
        for a in range(0, H, lignes):
            b = min(a + lignes, H)
            r, c = np.divmod(np.arange(a * W, b * W), W)
            ho, mo = self(r, c)
            hor[..., a:b, :] = np.moveaxis(ho.reshape(b - a, W, *ho.shape[1:]), (0, 1), (-2, -1))
            moy[..., a:b, :] = np.moveaxis(mo.reshape(b - a, W, *mo.shape[1:]), (0, 1), (-2, -1))
        return hor, moy, time.time() - t0


def horizons(s, cfg, rows, cols):
    return Horizons(s, cfg)(rows, cols)


# =============================================================================
# Étape 3 — Contrôle : coût et erreur contre l'horizon exact (descripteurs de B1), par hauteur
# =============================================================================
def main() -> None:
    import pandas as pd

    from src.baselines_surrogate import BANDES_M, _descripteurs, offsets_bandes
    from src.site import ROOT, load_config
    from src.viewshed import prepare_surfaces

    cfg = load_config()
    hc = cfg["surrogate"]["horizons"]
    assert tuple(hc["bandes_m"]) == BANDES_M, "mêmes bandes que B1 pour comparer"
    s = prepare_surfaces(cfg)
    hz = Horizons(s, cfg)
    print(f"{hz.blocs[0].size:,} blocs lus par position (calcul exact : 70 680 cellules), "
          f"{hz.h.size} hauteurs")

    # Bloc contigu du site (120 × 120), comme dans une carte dense : l'ordre des positions
    # compte (réutilisation du cache), un tirage aléatoire sous-estimerait le gain.
    r, c = np.mgrid[400:520, 400:520]
    r, c = r.ravel().astype(np.int64), c.ravel().astype(np.int64)
    hz(r[:10], c[:10])                                     # compilation
    t0 = time.time()
    P, _ = hz(r, c)
    tp = (time.time() - t0) / r.size

    dr, dc, sect, bande, dd = offsets_bandes(cfg)
    sol = s.z_opaque < s.z_tgt
    ns, nb = hz.ns, hz.nb
    h64 = hz.h.astype(np.float64)
    _descripteurs(s.z_opaque, s.z_veg, s.k_ext, sol, r[:10], c[:10], h64, dr, dc, sect, bande,
                  dd, ns, nb)
    t0 = time.time()
    E = _descripteurs(s.z_opaque, s.z_veg, s.k_ext, sol, r, c, h64, dr, dc, sect, bande, dd, ns, nb)
    te = (time.time() - t0) / r.size
    E = E.reshape(r.size, h64.size, ns, nb, 4)             # (n, h, s, b, desc)
    ex_o = np.arctan(E[..., 0]).transpose(0, 1, 3, 2)      # (n, h, b, s)

    rows = []
    for ih in (0, h64.size - 1):
        for b in range(nb):
            e = P[:, ih, 0, b] - ex_o[:, ih, b]
            rows.append(dict(h=h64[ih], bande=f"{BANDES_M[b]:g}-{BANDES_M[b + 1]:g} m",
                             pas_px=hc["pas_px"][b], biais_deg=np.degrees(e.mean()),
                             eam_deg=np.degrees(np.abs(e).mean()),
                             p95_deg=np.degrees(np.quantile(np.abs(e), 0.95)),
                             part_surestime=np.mean(e > 1e-6)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"Coût ({hz.h.size} hauteurs) : pyramide {1e3 * tp:.4f} ms par position "
          f"(+ {hz.t_prep:.1f} s de filtres, une fois pour la grille) ; exact (B1) {1e3 * te:.3f} ms "
          f"-> ×{te / tp:.0f}")


if __name__ == "__main__":
    main()
