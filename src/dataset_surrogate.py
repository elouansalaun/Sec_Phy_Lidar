"""
src/dataset_surrogate.py — Jeu de données exact du modèle de substitution (étape 5.2).

Cible (docs/formulation.md, §8.2) : pour un observateur o à la hauteur h au-dessus de la surface
opaque, part de la surface de chaque (anneau, secteur) qui est du sol visible, pondérée par
la transmittance T = exp(-τ) :

    V[h, a, s](o) = Σ_{j ∈ sol ∩ bin(a, s)} T_oj  /  |bin(a, s)|        ∈ [0, 1]

τ vient du même noyau que src/viewshed.py (ray_tau) : les étiquettes sont exactes par construction.

Découpage spatial (§8.4) : test = emprise du site + marge ; tampon exclu autour ; validation =
tuiles tirées au hasard ; entraînement = le reste. Seuls les observateurs dont le disque de rayon R
tient dans la grille sont retenus.

Sorties (surrogate.out_dir) :
  etiquettes.npz   observateurs échantillonnés (entraînement + validation)
  test.npz         observateurs du bloc test, tous les `test_pas_px` pixels
  decoupage.tif    0 entraînement, 1 validation, 2 test, 255 exclu
  strates.tif      0 toit, 1 sol dégagé, 2 sol sous végétation
  decoupage.png    carte de contrôle

Usage : python -m src.dataset_surrogate            (jeu complet)
        python -m src.dataset_surrogate --pilote 200   (chronométrage seul, rien n'est écrit)
"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import numpy as np
from numba import njit, prange

from src.rasters import write_raster
from src.site import ROOT, load_config
from src.viewshed import prepare_surfaces, ray_tau, to_pixel

STRATES = ("toit", "sol_degage", "sol_vegetation")
SPLITS = {"entrainement": 0, "validation": 1, "test": 2}
EXCLU = 255


# =============================================================================
# Étape 1 — Géométrie des bins : offsets du disque, anneau et secteur de chacun
# =============================================================================
def bins_disque(rayon_m: float, anneaux_m, n_secteurs: int, res: float):
    """Offsets (dr, dc) du disque (centre exclu), indice de bin = anneau·n_secteurs + secteur,
    et nombre de cellules par bin (dénominateur de V, indépendant du terrain).

    Azimut compté depuis le nord, dans le sens horaire (convention des caméras de src.sensors) ;
    le secteur k est centré sur k·360/n_secteurs."""
    R = int(rayon_m / res)
    dr, dc = np.mgrid[-R:R + 1, -R:R + 1]
    dr, dc = dr.ravel(), dc.ravel()
    d = np.hypot(dr, dc) * res
    keep = (dr * dr + dc * dc <= R * R) & (d > 0)              # même disque que optical_depth
    dr, dc, d = dr[keep], dc[keep], d[keep]
    az = np.degrees(np.arctan2(dc, -dr)) % 360.0               # est = +c, nord = −r
    w = 360.0 / n_secteurs
    sect = np.floor((az + w / 2) / w).astype(np.int64) % n_secteurs
    edges = np.asarray(anneaux_m, float)
    ann = np.searchsorted(edges[1:-1], d, side="left")          # d ≤ 50 -> 0 ; 50 < d ≤ 150 -> 1
    b = ann * n_secteurs + sect
    n_bin = np.bincount(b, minlength=(len(edges) - 1) * n_secteurs).astype(np.float64)
    return dr.astype(np.int64), dc.astype(np.int64), b, n_bin


# =============================================================================
# Étape 2 — Noyau : potentiel exact d'une liste d'observateurs
# =============================================================================
@njit(parallel=True, cache=True)
def potentiel(z_opaque, z_veg, k_ext, z_tgt, sol, obs_r, obs_c, hauteurs,
              dr, dc, b, n_bin, res):
    """V[i, h, bin] pour chaque observateur i. Parallèle sur les observateurs."""
    n, nh, nb = obs_r.size, hauteurs.size, n_bin.size
    out = np.zeros((n, nh, nb), np.float32)
    for i in prange(n):
        r0, c0 = obs_r[i], obs_c[i]
        acc = np.zeros(nb)
        for ih in range(nh):
            z_obs = z_opaque[r0, c0] + hauteurs[ih]
            acc[:] = 0.0
            for m in range(dr.size):
                r, c = r0 + dr[m], c0 + dc[m]
                if not sol[r, c]:
                    continue
                tau = ray_tau(z_opaque, z_veg, k_ext, r0, c0, z_obs, r, c, z_tgt[r, c], res)
                if np.isfinite(tau):
                    acc[b[m]] += np.exp(-tau)
            for q in range(nb):
                out[i, ih, q] = acc[q] / n_bin[q]
    return out


# =============================================================================
# Étape 3 — Strates et découpage spatial
# =============================================================================
def strates(s, cfg) -> np.ndarray:
    """0 toit (surface opaque au-dessus de la cible), 1 sol dégagé, 2 sol sous végétation."""
    sol = s.z_opaque < s.z_tgt
    veg = (s.z_veg - s.z_opaque) > cfg["viewshed"]["veg_min_h_m"]
    st = np.zeros(sol.shape, np.uint8)
    st[sol & ~veg] = 1
    st[sol & veg] = 2
    return st


def decoupage(s, cfg, site_bounds) -> np.ndarray:
    sc, g = cfg["surrogate"], s.grid
    H, W = s.z_tgt.shape
    R = int(sc["rayon_m"] / g.res)
    split = np.full((H, W), EXCLU, np.uint8)
    valide = np.zeros((H, W), bool)
    valide[R:H - R, R:W - R] = True                     # disque complet dans la grille

    # Test : emprise du site + marge (en pixels)
    xmin, ymin, xmax, ymax = site_bounds
    m = sc["test"]["marge_m"]
    r_a, c_a = to_pixel(g, xmin - m, ymax + m)
    r_b, c_b = to_pixel(g, xmax + m, ymin - m)
    test = np.zeros((H, W), bool)
    test[max(r_a, 0):r_b + 1, max(c_a, 0):c_b + 1] = True
    t = int(sc["test"]["tampon_m"] / g.res)
    tampon = np.zeros((H, W), bool)
    tampon[max(r_a - t, 0):r_b + t + 1, max(c_a - t, 0):c_b + t + 1] = True

    # Validation : tuiles carrées tirées au hasard hors tampon
    tu = int(sc["validation"]["tuile_m"] / g.res)
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    tr, tc = np.divmod(np.arange(H * W), W)
    tuile = (tr // tu) * (W // tu + 1) + (tc // tu)
    tuile = tuile.reshape(H, W)
    libres = np.unique(tuile[valide & ~tampon])
    n_val = int(round(sc["validation"]["part"] * libres.size))
    val_t = rng.choice(libres, n_val, replace=False)

    split[valide & ~tampon] = SPLITS["entrainement"]
    split[valide & ~tampon & np.isin(tuile, val_t)] = SPLITS["validation"]
    split[valide & test] = SPLITS["test"]
    return split


# =============================================================================
# Étape 4 — Échantillonnage des observateurs
# =============================================================================
def echantillonner(split, st, n: int, props: dict, rng):
    """Observateurs d'entraînement + validation, stratifiés par type de support."""
    rows, cols = [], []
    for k, name in enumerate(STRATES):
        cand = np.flatnonzero((split.ravel() <= SPLITS["validation"]) & (st.ravel() == k))
        m = min(int(round(n * props[name])), cand.size)
        pick = rng.choice(cand, m, replace=False)
        r, c = np.divmod(pick, split.shape[1])
        rows.append(r)
        cols.append(c)
    return np.concatenate(rows), np.concatenate(cols)


def observateurs_test(split, pas: int):
    r, c = np.nonzero(split == SPLITS["test"])
    keep = (r % pas == 0) & (c % pas == 0)
    return r[keep], c[keep]


# =============================================================================
# Étape 5 — Calcul par lots, avec suivi du temps
# =============================================================================
class Contexte:
    def __init__(self, cfg):
        self.cfg = cfg
        sc = cfg["surrogate"]
        self.s = prepare_surfaces(cfg)
        self.sol = self.s.z_opaque < self.s.z_tgt
        self.h = np.asarray(sc["hauteurs_m"], np.float64)
        self.dr, self.dc, self.b, self.n_bin = bins_disque(
            sc["rayon_m"], sc["anneaux_m"], sc["n_secteurs"], self.s.grid.res)
        self.forme = (self.h.size, len(sc["anneaux_m"]) - 1, sc["n_secteurs"])

    def calculer(self, rows, cols, lot: int = 2000, label: str = "") -> tuple[np.ndarray, float]:
        s = self.s
        out = np.empty((rows.size, self.h.size, self.n_bin.size), np.float32)
        t0 = time.time()
        for a in range(0, rows.size, lot):
            sl = slice(a, a + lot)
            out[sl] = potentiel(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, self.sol,
                                rows[sl].astype(np.int64), cols[sl].astype(np.int64), self.h,
                                self.dr, self.dc, self.b, self.n_bin, s.grid.res)
            done = min(a + lot, rows.size)
            el = time.time() - t0
            print(f"  {label} {done:>7,}/{rows.size:,}  {el:6.0f} s  "
                  f"(reste ≈ {el / done * (rows.size - done) / 60:.0f} min)", flush=True)
        return out.reshape(rows.size, *self.forme), time.time() - t0


def main() -> None:
    cfg = load_config()
    sc = cfg["surrogate"]
    pilote = int(sys.argv[sys.argv.index("--pilote") + 1]) if "--pilote" in sys.argv else 0
    ctx = Contexte(cfg)
    s = ctx.s
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    split = decoupage(s, cfg, per.bounds)
    st = strates(s, cfg)
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    n_valide = int((split != EXCLU).sum())
    cnt = {k: int((split == v).sum()) for k, v in SPLITS.items()}
    print(f"{ctx.dr.size:,} cellules par disque (R = {sc['rayon_m']} m), "
          f"{ctx.n_bin.size} bins × {ctx.h.size} hauteurs = {ctx.n_bin.size * ctx.h.size} sorties")
    print(f"Positions valides : {n_valide:,} ({cnt})")

    # Compilation hors chronométrage
    potentiel(s.z_opaque, s.z_veg, s.k_ext, s.z_tgt, ctx.sol, np.array([500]), np.array([500]),
              ctx.h, ctx.dr, ctx.dc, ctx.b, ctx.n_bin, s.grid.res)

    if pilote:
        r, c = echantillonner(split, st, pilote, sc["echantillonnage"]["proportions"], rng)
        _, dt = ctx.calculer(r, c, lot=max(pilote // 4, 1), label="pilote")
        per_obs = dt / r.size
        n_tr = sc["echantillonnage"]["n_observateurs"]
        n_te = observateurs_test(split, sc["test_pas_px"])[0].size
        print(f"\n{1e3 * per_obs:.1f} ms par observateur ({ctx.h.size} hauteurs) sur cette machine")
        print(f"  jeu d'entraînement ({n_tr:,}) : ≈ {per_obs * n_tr / 60:.0f} min")
        print(f"  bloc test ({n_te:,}, pas {sc['test_pas_px']} px) : ≈ {per_obs * n_te / 60:.0f} min")
        print(f"  carte exacte complète ({n_valide:,} positions) : ≈ {per_obs * n_valide / 3600:.1f} h")
        return

    out = ROOT / sc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_raster(out / "decoupage.tif", split, s.grid)
    write_raster(out / "strates.tif", st, s.grid)

    r, c = echantillonner(split, st, sc["echantillonnage"]["n_observateurs"],
                          sc["echantillonnage"]["proportions"], rng)
    Y, dt = ctx.calculer(r, c, label="entraînement+validation")
    np.savez_compressed(out / "etiquettes.npz", rows=r, cols=c, split=split[r, c], strate=st[r, c],
                        Y=Y, hauteurs=ctx.h, anneaux=np.asarray(sc["anneaux_m"]),
                        n_secteurs=sc["n_secteurs"], temps_s=dt)
    per_obs = dt / r.size

    rt, ct = observateurs_test(split, sc["test_pas_px"])
    Yt, dtt = ctx.calculer(rt, ct, label="test")
    np.savez_compressed(out / "test.npz", rows=rt, cols=ct, strate=st[rt, ct], Y=Yt,
                        hauteurs=ctx.h, anneaux=np.asarray(sc["anneaux_m"]),
                        n_secteurs=sc["n_secteurs"], temps_s=dtt)

    print(f"\n{r.size:,} observateurs échantillonnés en {dt / 60:.1f} min, "
          f"{rt.size:,} observateurs de test en {dtt / 60:.1f} min")
    print(f"Coût exact : {1e3 * per_obs:.1f} ms par observateur ; carte complète "
          f"({n_valide:,} positions) ≈ {per_obs * n_valide / 3600:.1f} h sur cette machine")
    for k, name in enumerate(STRATES):
        m = st[r, c] == k
        print(f"  {name:15s} n = {m.sum():6,}  V moyen (h = {ctx.h[-1]:g} m, anneau 50–150 m) = "
              f"{Y[m, -1, -1].mean():.3f}")
    figure(split, st, r, c, per, s.grid, out / "decoupage.png")
    print(f"-> {out}")


def figure(split, st, r, c, per, grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)
    img = np.where(split == EXCLU, np.nan, split.astype(float))
    ext = (grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    ax.imshow(img, cmap=ListedColormap(["#dfe9f5", "#f6dcc8", "#cfeede"]), extent=ext,
              vmin=0, vmax=2, interpolation="nearest")
    x = grid.xmin + (c + 0.5) * grid.res
    y = grid.ymax - (r + 0.5) * grid.res
    cols = np.array(["#4a3aa7", "#2a78d6", "#1baf7a"])[st[r, c]]
    ax.scatter(x, y, s=0.3, c=cols, lw=0)
    ax.plot(*per.exterior.xy, color="#0b0b0b", lw=1)
    ax.set_title("Découpage spatial (bleu : entraînement, orange : validation, vert : test)\n"
                 "points : observateurs (violet toit, bleu sol dégagé, vert sous végétation)",
                 loc="left", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
